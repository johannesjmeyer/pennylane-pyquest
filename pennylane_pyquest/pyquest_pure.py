# Copyright 2020 Johannes Jakob Meyer.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyQuest Pure state 
========

**Module name:** :mod:`pennylane_pyquest.pure`

.. currentmodule:: pennylane_pyquest.pure

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Pyquest device/simulator as a PennyLane device.

It can inherit from the abstract PyquestDevice to reduce
code duplication if needed.


See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of Device methods available.

Classes
-------

.. autosummary::
   PyquestPure

----
"""

# we always import NumPy directly
import numpy as np
from .pyquest_device import PyquestDevice
import pyquest_cffi as pqc


class QuregContext:
    def __init__(self, wires):
        self.wires = wires

    def __enter__(self):
        self.env = pqc.utils.createQuestEnv()()
        self.qureg = pqc.utils.createQureg()(self.wires, env=self.env)

        return self

    def __exit__(self, etype, value, traceback):
        pqc.utils.destroyQureg()(self.qureg, env=self.env)
        pqc.utils.destroyQuestEnv()(self.env)


class PyquestPure(PyquestDevice):
    operations = {
        "BasisState",
        "QubitStateVector",
        "QubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "PauliRot",
        "Hadamard",
        "S",
        "T",
        "CNOT",
        "SWAP",
        "CZ",
        "PhaseShift",
        "RX",
        "RY",
        "RZ",
        "CRX",
        "CRY",
        "CRZ",
    }

    def _qureg_context(self):
        return QuregContext(self.num_wires)

    def _get_state(self, context):
        self._state = pqc.cheat.getStateVector()(context.qureg)

    @property
    def state(self):
        return self._state
