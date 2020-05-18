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
from pennylane_pyquest import PyquestPure

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
