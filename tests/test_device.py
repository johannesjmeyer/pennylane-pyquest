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
import pytest

import numpy as np
from pennylane_pyquest.pyquest_device import PyquestDevice
import pennylane as qml


class TestAbstract:
    def test_apply(self):
        dev = PyquestDevice(wires=3)

        dev.apply(
            [
                qml.PauliRot(np.pi / 2, "IIX", wires=[0, 1, 2]),
                qml.PauliRot(np.pi / 2, "IXI", wires=[0, 1, 2]),
                qml.PauliRot(np.pi / 2, "XII", wires=[0, 1, 2]),
            ]
        )

        assert False
