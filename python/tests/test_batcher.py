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

"""Unit tests for triton_backend_utils.batcher."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

# Make the under-test package importable without installation.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _PKG_ROOT)

# We patch triton_python_backend_utils *before* importing our module so
# that the real backend does not need to be installed in this environment.
import triton_backend_utils.batcher as batcher_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles that mimic the triton_python_backend_utils runtime.
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Mimics triton_python_backend_utils.InferenceInput / Tensor."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr)

    def as_numpy(self) -> np.ndarray:
        return self._arr


class _FakeRequest:
    """Mimics triton_python_backend_utils.InferenceRequest."""

    def __init__(self, tensor_map: Dict[str, np.ndarray]) -> None:
        self._tensors = {name: _FakeTensor(arr) for name, arr in tensor_map.items()}

    def inputs(self) -> List:
        return [
            type("_In", (), {"name": n})() for n in self._tensors
        ]

    def __getitem__(self, key: str) -> _FakeTensor:
        return self._tensors[key]


class _FakeResponse:
    """Mimics triton_python_backend_utils.InferenceResponse."""

    def __init__(self, tensors: List) -> None:
        self.tensors = tensors


class _FakePbUtils:
    """Stand-in for the real triton_python_backend_utils module."""

    @staticmethod
    def get_input_tensor_by_name(request: Any, name: str) -> Any:
        try:
            return request[name]
        except (KeyError, TypeError):
            return None

    @staticmethod
    def Tensor(name: str, arr: np.ndarray) -> _FakeTensor:
        t = _FakeTensor(arr)
        t.name = name
        return t

    @staticmethod
    def InferenceResponse(tensors: List) -> _FakeResponse:
        return _FakeResponse(tensors)


# Swap the module-level reference used by batcher.py so split_outputs works.
batcher_mod.pb_utils = _FakePbUtils()


def _make_requests(n: int = 3, *, shape=(4,), dtype=np.float32, prefix="INPUT") -> List[_FakeRequest]:
    """Helper that produces n identical-shape fake requests."""
    return [
        _FakeRequest(
            {f"{prefix}0": np.full(shape, fill_value=i, dtype=dtype),
             f"{prefix}1": np.full(shape, fill_value=i + 10, dtype=dtype)}
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# BatchInputs basics
# ---------------------------------------------------------------------------


class TestBatchedInputs:
    def test_construct_and_access(self):
        arrays = {"X": np.zeros((3, 4), dtype=np.float32), "Y": np.ones((3, 2))}
        bi = batcher_mod.BatchedInputs(arrays, batch_size=3)
        assert bi.batch_size == 3
        assert bi.input_names == ["X", "Y"]
        assert bi["X"].shape == (3, 4)
        assert dict(bi) == arrays
        assert len(bi) == 2

    def test_mismatched_batch_size_raises(self):
        with pytest.raises(batcher_mod.BatchingError):
            batcher_mod.BatchedInputs({"X": np.zeros((5, 4))}, batch_size=3)


# ---------------------------------------------------------------------------
# stack_input_tensors
# ---------------------------------------------------------------------------


class TestStackInputTensors:
    def test_basic_stack_default_axis(self):
        requests = _make_requests(3, shape=(4,), dtype=np.float32)
        out = batcher_mod.stack_input_tensors(requests, ["INPUT0", "INPUT1"])
        assert out["INPUT0"].shape == (3, 4)
        assert out["INPUT1"].shape == (3, 4)
        np.testing.assert_array_equal(out["INPUT0"][0], np.zeros(4))
        np.testing.assert_array_equal(out["INPUT0"][1], np.full(4, 1.0))

    def test_autodiscover_input_names(self):
        requests = _make_requests(4)
        out = batcher_mod.stack_input_tensors(requests, input_names=None)
        assert set(out.keys()) == {"INPUT0", "INPUT1"}
        assert out["INPUT0"].shape == (4, 4)

    def test_empty_requests_raises(self):
        with pytest.raises(ValueError):
            batcher_mod.stack_input_tensors([], ["INPUT0"])

    def test_missing_input_raises(self):
        # Both have INPUT0 with same shape, but second is missing INPUT1.
        reqs = [
            _FakeRequest({"INPUT0": np.zeros(4, dtype=np.float32),
                          "INPUT1": np.zeros(4, dtype=np.float32)}),
            _FakeRequest({"INPUT0": np.ones(4, dtype=np.float32)}),
        ]
        with pytest.raises(batcher_mod.MissingInputError):
            batcher_mod.stack_input_tensors(reqs, ["INPUT0", "INPUT1"])

    def test_inconsistent_shape_raises(self):
        reqs = [
            _FakeRequest({"X": np.zeros((3, 4), dtype=np.float32)}),
            _FakeRequest({"X": np.zeros((3, 5), dtype=np.float32)}),
        ]
        with pytest.raises(batcher_mod.InconsistentShapeError):
            batcher_mod.stack_input_tensors(reqs, ["X"])

    def test_inconsistent_shape_allowed(self):
        reqs = [
            _FakeRequest({"X": np.zeros((3, 4), dtype=np.float32)}),
            _FakeRequest({"X": np.zeros((3, 4), dtype=np.float32)}),
        ]
        out = batcher_mod.stack_input_tensors(reqs, ["X"], require_same_shape=False)
        assert out["X"].shape == (2, 3, 4)

    def test_inconsistent_dtype_raises(self):
        reqs = [
            _FakeRequest({"X": np.zeros(4, dtype=np.float32)}),
            _FakeRequest({"X": np.zeros(4, dtype=np.int32)}),
        ]
        with pytest.raises(batcher_mod.InconsistentDtypeError):
            batcher_mod.stack_input_tensors(reqs, ["X"])

    def test_stack_along_nonzero_axis(self):
        reqs = [
            _FakeRequest({"X": np.zeros((2, 3), dtype=np.float32)}),
            _FakeRequest({"X": np.ones((2, 3), dtype=np.float32)}),
        ]
        out = batcher_mod.stack_input_tensors(reqs, ["X"], stack_axis=1)
        assert out["X"].shape == (2, 2, 3)  # batch inserted at axis 1

    def test_high_dimensional_tensors(self):
        requests = _make_requests(2, shape=(3, 4, 5), dtype=np.float64)
        out = batcher_mod.stack_input_tensors(requests, ["INPUT0"])
        assert out["INPUT0"].shape == (2, 3, 4, 5)
        assert out["INPUT0"].dtype == np.float64

    def test_various_dtypes(self):
        for dt in [np.float16, np.float32, np.float64, np.int8, np.int16,
                    np.int32, np.int64, np.uint8, np.bool_, np.complex64]:
            reqs = [_FakeRequest({"X": np.zeros(4, dtype=dt)}) for _ in range(2)]
            out = batcher_mod.stack_input_tensors(reqs, ["X"])
            assert out["X"].dtype == dt
            assert out["X"].shape == (2, 4)

    def test_string_arrays(self):
        reqs = [
            _FakeRequest({"X": np.array(["abc", "def"], dtype=np.bytes_)}),
            _FakeRequest({"X": np.array(["ghi", "jkl"], dtype=np.bytes_)}),
        ]
        out = batcher_mod.stack_input_tensors(reqs, ["X"])
        assert out["X"].shape == (2, 2)
        assert out["X"][0, 0] == b"abc"

    def test_single_request(self):
        reqs = [_make_requests(1)[0]]
        out = batcher_mod.stack_input_tensors(reqs, ["INPUT0"])
        # Even with one request we still prepend the batch axis.
        assert out["INPUT0"].shape[0] == 1


# ---------------------------------------------------------------------------
# batch_inputs
# ---------------------------------------------------------------------------


class TestBatchInputs:
    def test_returns_batchedinputs(self):
        requests = _make_requests(5)
        bi = batcher_mod.batch_inputs(requests)
        assert isinstance(bi, batcher_mod.BatchedInputs)
        assert bi.batch_size == 5
        assert bi["INPUT0"].shape == (5, 4)

    def test_round_trip_split(self):
        requests = _make_requests(3, shape=(6,))
        bi = batcher_mod.batch_inputs(requests)
        # Simulate model output:  batched sum
        batched_out = bi["INPUT0"] + bi["INPUT1"]  # (3, 6)
        per_request = batcher_mod.split_output_arrays(batched_out, 3, ["OUTPUT"])
        assert len(per_request) == 3
        for i, d in enumerate(per_request):
            expected = np.full(6, i, dtype=np.float32) + np.full(6, i + 10, dtype=np.float32)
            np.testing.assert_array_equal(d["OUTPUT"], expected)


# ---------------------------------------------------------------------------
# split_output_arrays
# ---------------------------------------------------------------------------


class TestSplitOutputArrays:
    def test_split_plain_array(self):
        arr = np.arange(12).reshape(3, 4)
        per = batcher_mod.split_output_arrays(arr, 3, ["OUT"])
        assert len(per) == 3
        np.testing.assert_array_equal(per[0]["OUT"], [0, 1, 2, 3])

    def test_split_mapping(self):
        batched = {"A": np.zeros((3, 4)), "B": np.ones((3, 2))}
        per = batcher_mod.split_output_arrays(batched, 3)
        assert per[0]["A"].shape == (4,)
        assert per[0]["B"].shape == (2,)

    def test_split_sequence(self):
        seq = [np.zeros((3, 4)), np.ones((3, 2))]
        per = batcher_mod.split_output_arrays(seq, 3, ["X", "Y"])
        assert per[0]["X"].shape == (4,)
        assert per[0]["Y"].shape == (2,)

    def test_bad_sequence_length_raises(self):
        with pytest.raises(ValueError):
            batcher_mod.split_output_arrays([np.zeros((3, 4)), np.zeros((3, 2))], 3, ["A"])

    def test_missing_names_for_plain_array(self):
        with pytest.raises(ValueError):
            batcher_mod.split_output_arrays(np.zeros((3, 4)), 3)

    def test_batch_mismatch_raises(self):
        arr = np.zeros((5, 4))
        with pytest.raises(batcher_mod.BatchingError):
            batcher_mod.split_output_arrays(arr, 3, ["OUT"])

    def test_split_along_axis_one(self):
        arr = np.arange(24).reshape(4, 3, 2)
        per = batcher_mod.split_output_arrays(arr, 3, ["OUT"], split_axis=1)
        assert len(per) == 3
        assert per[0]["OUT"].shape == (4, 2)

    def test_singleton_batch(self):
        arr = np.zeros((1, 4, 5))
        per = batcher_mod.split_output_arrays(arr, 1, ["OUT"])
        assert per[0]["OUT"].shape == (4, 5)

    def test_zero_batch_raises(self):
        with pytest.raises(ValueError):
            batcher_mod.split_output_arrays(np.zeros((0, 4)), 0, ["OUT"])


# ---------------------------------------------------------------------------
# split_outputs (requires fake pb_utils)
# ---------------------------------------------------------------------------


class TestSplitOutputs:
    def test_returns_inference_responses(self):
        requests = _make_requests(3)
        batched = {"OUT0": np.zeros((3, 4), dtype=np.float32),
                    "OUT1": np.ones((3, 4), dtype=np.float32)}
        responses = batcher_mod.split_outputs(batched, requests, ["OUT0", "OUT1"])
        assert len(responses) == 3
        for resp in responses:
            assert isinstance(resp, _FakeResponse)
            assert len(resp.tensors) == 2

    def test_split_outputs_plain_array(self):
        requests = _make_requests(3)
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        responses = batcher_mod.split_outputs(arr, requests, ["OUT"])
        assert responses[0].tensors[0].as_numpy().tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_split_outputs_sequence(self):
        requests = _make_requests(4)
        seq = [np.zeros((4, 5)), np.ones((4, 2))]
        responses = batcher_mod.split_outputs(seq, requests, ["A", "B"])
        assert len(responses) == 4
        assert responses[0].tensors[0].as_numpy().shape == (5,)
        assert responses[0].tensors[1].as_numpy().shape == (2,)


# ---------------------------------------------------------------------------
# End-to-end: mimic a real execute() flow.
# ---------------------------------------------------------------------------


class TestEndToEndExecute:
    def test_identity_model(self):
        requests = _make_requests(3, shape=(8,), dtype=np.float32)

        # Fake execute() implementation using the helpers.
        bi = batcher_mod.batch_inputs(requests)
        batched_out = bi["INPUT0"]  # identity
        responses = batcher_mod.split_outputs(batched_out, requests, ["OUTPUT0"])

        assert len(responses) == 3
        for i, resp in enumerate(responses):
            out_arr = resp.tensors[0].as_numpy()
            np.testing.assert_array_equal(out_arr, np.full(8, i, dtype=np.float32))

    def test_add_sub_model(self):
        requests = _make_requests(4, shape=(6,), dtype=np.float32)
        bi = batcher_mod.batch_inputs(requests)
        out_add = bi["INPUT0"] + bi["INPUT1"]
        out_sub = bi["INPUT0"] - bi["INPUT1"]
        responses = batcher_mod.split_outputs(
            {"OUTPUT0": out_add, "OUTPUT1": out_sub},
            requests, ["OUTPUT0", "OUTPUT1"],
        )
        assert len(responses) == 4
        for i, resp in enumerate(responses):
            in0_val = float(i)
            in1_val = float(i + 10)
            add_arr = resp.tensors[0].as_numpy()
            sub_arr = resp.tensors[1].as_numpy()
            np.testing.assert_array_equal(add_arr, np.full(6, in0_val + in1_val))
            np.testing.assert_array_equal(sub_arr, np.full(6, in0_val - in1_val))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
