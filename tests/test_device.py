# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for any plugin- or framework-specific behaviour of the plugin devices"""
import numpy as np
import pennylane as qml
import pytest

import pennylane_pyquest
from pennylane_pyquest import PyquestPure, PyquestMixed

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ],
    dtype=np.complex,
)


class TestAbstract:
    def no_test_apply(self):
        dev = PyquestPure(wires=2)

        dev.apply(
            [
                qml.QubitUnitary(U, wires=[0]),
                # qml.BasisState(np.array([0, 1]), wires=[0, 1]),
                # qml.PauliX(0),
                # qml.PauliX(1),
                # qml.CNOT(wires=[0, 1])
            ]
        )

        # assert False

def simple_error_model(operation):
    if operation.num_wires == 1:
        return [pennylane_pyquest.ops.MixDephasing(0.01, wires=operation.wires)]
        
    return [pennylane_pyquest.ops.MixDephasing(0.03, wires=w) for w in operation.wires]

class TestErrorModel:

    def test_error_model(self):
        dev = PyquestMixed(wires=3, error_model=simple_error_model)

        res = dev._preprocess_operations([
            qml.Hadamard(0),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(0.54, wires=[0]),
            qml.CNOT(wires=[1, 2]),
        ])

        assert res[0].name == "Hadamard"
        assert res[1].name == "MixDephasing"
        assert res[2].name == "CNOT"
        assert res[3].name == "MixDephasing"
        assert res[4].name == "MixDephasing"
        assert res[5].name == "Hadamard"
        assert res[6].name == "MixDephasing"
        assert res[7].name == "CNOT"
        assert res[8].name == "MixDephasing"
        assert res[9].name == "MixDephasing"

        assert False

    def test_error_model(self):
        err_dev = PyquestMixed(wires=3, error_model=simple_error_model)
        dev = PyquestMixed(wires=3)

        def circuit():
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.Hadamard(2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.RY(0.54, wires=[0])
            qml.RY(0.66, wires=[1])
            qml.RY(0.98, wires=[2])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, dev)
        err_node = qml.QNode(circuit, err_dev)

        print(node())
        print(err_node())

        assert node() != err_node()