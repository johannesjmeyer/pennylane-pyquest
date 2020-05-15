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
Device 1
========

**Module name:** :mod:`pennylane_pyquest.PyquestMixed`

.. currentmodule:: pennylane_pyquest.PyquestMixed

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Pyquest device/simulator as a PennyLane device.

It can inherit from the abstract PyquestDevice to reduce
code duplication if needed.


See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of Device methods available.

Classes
-------

.. autosummary::
   PyquestMixed

----
"""
import numpy as np
from .pyquest_device import PyquestDevice
import pyquest_cffi as pqc


class DensityQuregContext:
    def __init__(self, wires):
        self.wires = wires

    def __enter__(self):
        self.env = pqc.utils.createQuestEnv()()
        self.qureg = pqc.utils.createDensityQureg()(self.wires, env=self.env)

        return self

    def __exit__(self, etype, value, traceback):
        pqc.utils.destroyQureg()(self.qureg, env=self.env)
        pqc.utils.destroyQuestEnv()(self.env)


class PyquestMixed(PyquestDevice):

    operations = {
        "BasisState",
        "QubitStateVector",
        # "QubitUnitary", # Theoretically supportable, but silently crashes due to C errors
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
        "MixDephasing",
        "MixDepolarising",
        "MixDamping",
        "MixKrausMap",
    }

    def _qureg_context(self):
        return DensityQuregContext(self.num_wires)

    def _extract_information(self, context):
        self._density_matrix = pqc.cheat.getDensityMatrix()(context.qureg)
        self._probs = np.diag(self._density_matrix)

    @property
    def state(self):
        return None

    @property
    def density_matrix(self):
        return self._density_matrix
