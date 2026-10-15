# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Core batching helpers for the Triton Python backend.

The Triton Python backend's ``execute()`` receives a flat list of
``InferenceRequest`` objects.  Each request usually carries the same set
of named input tensors but with a leading dimension that differs per
user (batch size 1 from the client's point of view).  The helpers below
collect those tensors, ``np.stack`` them into a single batched array
along the first axis, and split the model's batched output back into
per-request responses.

All functions in this module are **pure** with respect to the request /
tensor objects: they only call the documented
``get_input_tensor_by_name`` / ``as_numpy`` API and build standard numpy
arrays.  They therefore work identically with the real
``triton_python_backend_utils`` runtime and with any test-double
implementations used in unit tests.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import triton_python_backend_utils as pb_utils  # type: ignore
except ImportError:  # pragma: no cover - runtime only
    pb_utils = None  # type: ignore


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BatchingError(Exception):
    """Base class for batching-related errors."""


class InconsistentShapeError(BatchingError):
    """Raised when input tensors for the same name have different shapes."""


class InconsistentDtypeError(BatchingError):
    """Raised when input tensors for the same name have different dtypes."""


class MissingInputError(BatchingError):
    """Raised when an expected input tensor is missing from a request."""


# ---------------------------------------------------------------------------
# BatchedInputs container
# ---------------------------------------------------------------------------


class BatchedInputs(Mapping[str, np.ndarray]):
    """Immutable mapping-like object that holds stacked input arrays.

    ``BatchedInputs`` behaves like a ``dict[str, np.ndarray]`` where every
    array has a leading batch dimension equal to the number of original
    requests.  It also exposes some convenience attributes for users who
    need to introspect the batch (e.g. for logging or reshaping).

    Attributes
    ----------
    batch_size : int
        Number of requests that contributed to this batch.
    input_names : list[str]
        Ordered list of input names present in the batch.
    shapes : dict[str, tuple[int, ...]]
        Shape of each batched array (including the leading batch dim).
    """

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        batch_size: int,
    ) -> None:
        self._arrays: Dict[str, np.ndarray] = dict(arrays)
        self._batch_size = int(batch_size)
        # Sanity check: every array's first dim must equal batch_size.
        for name, arr in self._arrays.items():
            if arr.shape[0] != self._batch_size:
                raise BatchingError(
                    f"Batched array for '{name}' has first dimension "
                    f"{arr.shape[0]}, expected {self._batch_size}."
                )

    # Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[key]

    def __iter__(self):
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    # Convenience accessors -------------------------------------------
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def input_names(self) -> List[str]:
        return list(self._arrays.keys())

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {name: arr.shape for name, arr in self._arrays.items()}

    def __repr__(self) -> str:
        items = ", ".join(
            f"{k!r}: shape={v.shape}, dtype={v.dtype}"
            for k, v in self._arrays.items()
        )
        return f"BatchedInputs(batch_size={self._batch_size}, {items})"


# ---------------------------------------------------------------------------
# Tensor extraction helpers (pure-python, pb_utils-free)
# ---------------------------------------------------------------------------


def _get_request_input_names(request: Any) -> List[str]:
    """Best-effort extraction of all input tensor names from a request.

    Different versions of ``triton_python_backend_utils`` expose different
    accessors; this helper tries the common ones and falls back to an
    introspection of ``dir(request)``.
    """
    # 1. Modern API: request.inputs() returns a list of InferenceInput.
    try:
        inputs = request.inputs()
        return [getattr(t, "name", None) or "" for t in inputs if getattr(t, "name", None)]
    except Exception:
        pass

    # 2. Try iterating the request directly (old API exposed a dict-like).
    try:
        return list(request.keys())
    except Exception:
        pass

    # 3. Fallback: look for common attribute names via dir().
    names = []
    for attr in dir(request):
        if attr.startswith("_"):
            continue
        if attr in {"inputs", "request_id"}:
            continue
        # Heuristic: anything that starts with INPUT or is all-caps.
        if attr.isupper() or attr.startswith("input"):
            names.append(attr)
    return names


def _get_input_tensor(request: Any, name: str) -> Any:
    """Return the input tensor object for ``name`` from ``request``.

    Prefers ``pb_utils.get_input_tensor_by_name`` when available, then
    falls back to direct attribute / dict access.  Returns ``None`` when
    the input cannot be found.
    """
    # 1. Official API.
    if pb_utils is not None:
        getter = getattr(pb_utils, "get_input_tensor_by_name", None)
        if getter is not None:
            tensor = getter(request, name)
            if tensor is not None:
                return tensor

    # 2. Dict-like access.
    try:
        tensor = request[name]
        if tensor is not None:
            return tensor
    except (KeyError, TypeError, IndexError):
        pass

    # 3. Attribute access.
    tensor = getattr(request, name, None)
    if tensor is not None:
        return tensor

    return None


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Convert an input tensor object to a numpy array.

    Handles the canonical ``as_numpy()`` method as well as fallbacks
    (``to_numpy()``, direct ``np.asarray`` of a bare array).
    """
    if tensor is None:
        raise ValueError("Cannot convert None tensor to numpy.")

    # Triton tensor objects expose .as_numpy().
    as_numpy = getattr(tensor, "as_numpy", None)
    if callable(as_numpy):
        arr = as_numpy()
        # Some older versions returned a list; normalise.
        return np.asarray(arr)

    to_numpy = getattr(tensor, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())

    # Last resort: treat the value itself as array-like.
    return np.asarray(tensor)


# ---------------------------------------------------------------------------
# Public API: stacking inputs
# ---------------------------------------------------------------------------


def stack_input_tensors(
    requests: Sequence[Any],
    input_names: Sequence[str],
    stack_axis: int = 0,
    require_same_shape: bool = True,
    require_same_dtype: bool = True,
) -> Dict[str, np.ndarray]:
    """Stack the named inputs of every request into batched numpy arrays.

    This is the lower-level building block used by :func:`batch_inputs`.
    Callers that don't need the :class:`BatchedInputs` wrapper or the
    output-splitting helpers can use it directly.

    Parameters
    ----------
    requests : sequence
        ``InferenceRequest`` objects as received by ``execute()``.
    input_names : sequence of str
        Names of the input tensors to collect.  If ``None``, every input
        name present in **all** requests is used.
    stack_axis : int, default 0
        Axis along which to stack.  The default prepends a new batch
        dimension, which is what most batched PyTorch / TensorFlow / ONNX
        models expect.
    require_same_shape : bool, default True
        When ``True`` (recommended) inputs whose shapes differ across
        requests raise :class:`InconsistentShapeError`.  Set to ``False``
        to allow ragged inputs – in that case ``np.stack`` will raise by
        itself if shapes are actually incompatible.
    require_same_dtype : bool, default True
        When ``True`` (recommended) inputs whose dtypes differ across
        requests raise :class:`InconsistentDtypeError`.  When ``False``
        numpy's usual promotion rules apply.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from input name to a stacked numpy array.  The shape of
        each returned array has one extra leading dimension of size
        ``len(requests)`` (when ``stack_axis == 0``).
    """
    if not requests:
        raise ValueError("`requests` must be a non-empty sequence.")

    requests = list(requests)

    # Auto-discover input names when not provided.
    if input_names is None:
        common_names: Optional[set] = None
        for req in requests:
            names = set(_get_request_input_names(req))
            common_names = names if common_names is None else common_names & names
        input_names = sorted(common_names or [])
        if not input_names:
            raise ValueError(
                "No common input names found across requests and no "
                "`input_names` argument was supplied."
            )

    input_names = list(input_names)
    num_requests = len(requests)
    stacked: Dict[str, np.ndarray] = {}

    for name in input_names:
        per_request: List[np.ndarray] = []
        for i, req in enumerate(requests):
            tensor = _get_input_tensor(req, name)
            if tensor is None:
                raise MissingInputError(
                    f"Request {i} is missing input tensor '{name}'."
                )
            arr = _tensor_to_numpy(tensor)
            per_request.append(arr)

        # Optional validation.
        if require_same_shape:
            shapes = {arr.shape for arr in per_request}
            if len(shapes) > 1:
                raise InconsistentShapeError(
                    f"Input '{name}' has inconsistent shapes across "
                    f"requests: {sorted(shapes)}."
                )

        if require_same_dtype:
            dtypes = {arr.dtype for arr in per_request}
            if len(dtypes) > 1:
                raise InconsistentDtypeError(
                    f"Input '{name}' has inconsistent dtypes across "
                    f"requests: {sorted(dtypes)}."
                )

        stacked[name] = np.stack(per_request, axis=stack_axis)

    return stacked


def batch_inputs(
    requests: Sequence[Any],
    input_names: Optional[Sequence[str]] = None,
    stack_axis: int = 0,
    require_same_shape: bool = True,
    require_same_dtype: bool = True,
) -> BatchedInputs:
    """Collect input tensors from every request and return them stacked.

    This is the high-level entry point.  See :func:`stack_input_tensors`
    for parameter details; this function simply wraps the result in a
    :class:`BatchedInputs` container.

    Typical usage from inside ``execute()``::

        def execute(self, requests):
            batched = batch_inputs(requests)
            y = self.model(batched["INPUT0"], batched["INPUT1"])   # batched forward pass
            responses = split_outputs(y, requests, ["OUTPUT0", "OUTPUT1"])
            return responses
    """
    arrays = stack_input_tensors(
        requests=requests,
        input_names=input_names,
        stack_axis=stack_axis,
        require_same_shape=require_same_shape,
        require_same_dtype=require_same_dtype,
    )
    return BatchedInputs(arrays, batch_size=len(requests))


# ---------------------------------------------------------------------------
# Public API: splitting outputs back into per-request responses
# ---------------------------------------------------------------------------


def split_output_arrays(
    batched_output: Union[np.ndarray, Mapping[str, np.ndarray], Sequence[np.ndarray]],
    num_requests: int,
    output_names: Optional[Sequence[str]] = None,
    split_axis: int = 0,
) -> List[Dict[str, np.ndarray]]:
    """Split batched output arrays into per-request dictionaries.

    Parameters
    ----------
    batched_output : np.ndarray, dict, or sequence
        Batched model output(s).  A single array is treated as the output
        for ``output_names[0]``; a mapping is keyed by output name; a
        sequence is matched positionally against ``output_names``.
    num_requests : int
        Number of requests in the original batch.  Used to validate that
        every batched array's ``split_axis`` dimension equals
        ``num_requests``.
    output_names : sequence of str, optional
        Names to use when ``batched_output`` is a plain array or
        sequence.  Required in those cases.
    split_axis : int, default 0
        Axis along which to split.  Must match the ``stack_axis`` used
        when batching inputs (usually 0).

    Returns
    -------
    list[dict[str, np.ndarray]]
        One dict per request, mapping output name to a single-request
        numpy array (the leading batch dimension has been removed).
    """
    if num_requests <= 0:
        raise ValueError("`num_requests` must be a positive integer.")

    # Normalise into {name: array}.
    if isinstance(batched_output, np.ndarray):
        if not output_names:
            raise ValueError(
                "`output_names` is required when `batched_output` is a "
                "plain numpy array."
            )
        outputs: Dict[str, np.ndarray] = {output_names[0]: batched_output}
    elif isinstance(batched_output, Mapping):
        outputs = dict(batched_output)
    elif isinstance(batched_output, Sequence):
        if not output_names or len(output_names) != len(batched_output):
            raise ValueError(
                "`output_names` must be provided and have the same length "
                "as `batched_output` when it is a sequence."
            )
        outputs = dict(zip(output_names, batched_output))
    else:
        raise TypeError(
            f"Unsupported type for `batched_output`: {type(batched_output)!r}. "
            "Expected np.ndarray, mapping, or sequence of arrays."
        )

    # Validate and split each output.
    per_request: List[Dict[str, np.ndarray]] = [{} for _ in range(num_requests)]
    for name, arr in outputs.items():
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        if arr.shape[split_axis] != num_requests:
            raise BatchingError(
                f"Output '{name}' has size {arr.shape[split_axis]} along "
                f"split axis {split_axis}, expected {num_requests}."
            )

        # np.split along the requested axis.
        slices = np.split(arr, num_requests, axis=split_axis)
        for i, sl in enumerate(slices):
            # Squeeze out the split axis (size is always 1 per slice).
            per_request[i][name] = np.squeeze(sl, axis=split_axis)

    return per_request


def split_outputs(
    batched_output: Union[np.ndarray, Mapping[str, np.ndarray], Sequence[np.ndarray]],
    requests: Sequence[Any],
    output_names: Sequence[str],
    split_axis: int = 0,
) -> List[Any]:
    """Split a batched model output into ``InferenceResponse`` objects.

    This is the inverse of :func:`batch_inputs` and returns exactly the
    ``List[InferenceResponse]`` that Python backend models must return
    from ``execute()``.

    Parameters
    ----------
    batched_output : np.ndarray, dict, or sequence
        Model output(s) produced from the stacked inputs.
    requests : sequence
        The same ``InferenceRequest`` sequence that was passed to
        :func:`batch_inputs`.  Only its length is consulted.
    output_names : sequence of str
        Names to use for each output tensor.
    split_axis : int, default 0
        Axis along which to split.  Should match the ``stack_axis``
        passed to :func:`batch_inputs`.

    Returns
    -------
    list[pb_utils.InferenceResponse]
        One response per original request, each carrying the per-request
        output tensors.

    Raises
    ------
    ImportError
        If ``triton_python_backend_utils`` is not importable (outside the
        Triton runtime).
    """
    if pb_utils is None:
        raise ImportError(
            "`split_outputs` requires `triton_python_backend_utils` to be "
            "available.  Use `split_output_arrays` in pure-Python contexts."
        )

    per_request = split_output_arrays(
        batched_output=batched_output,
        num_requests=len(requests),
        output_names=output_names,
        split_axis=split_axis,
    )

    responses = []
    for per_req_dict in per_request:
        tensors = [
            pb_utils.Tensor(name, arr) for name, arr in per_req_dict.items()
        ]
        responses.append(pb_utils.InferenceResponse(tensors))
    return responses
