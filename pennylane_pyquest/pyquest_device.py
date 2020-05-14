# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Pyquest device class
===========================

**Module name:** :mod:`pennylane_pyquest.device`

.. currentmodule:: pennylane_pyquest.device

An abstract base class for constructing Pyquest devices for PennyLane.

Classes
-------

.. autosummary::
   PyquestDevice

Code details
~~~~~~~~~~~~
"""
import abc
import itertools

# we always import NumPy directly
import numpy as np
from pennylane import QubitDevice
import pyquest_cffi as pqc
from ._version import __version__

class QuregContext:
    def __init__(self, wires):
        self.wires = wires

    def __enter__(self):
        self.env = pqc.utils.createQuestEnv()()
        self.qureg = pqc.utils.createQureg()(wires, env=env)

        return self

    def __exit__():
        pqc.utils.destroyQureg(self.qureg, env=self.env)
        pqc.utils.destroyQuestEng(self.env)


class PyquestDevice(QubitDevice):
    r"""Abstract Pyquest device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
        additional_option (float): as many additional arguments can be
            added as needed
    """
    name = 'Pyquest Simulator PennyLane plugin'
    pennylane_requires = '>=0.8.0'
    version = __version__
    author = 'Johannes Jakob Meyer'

    short_name = 'pyquest.base'
    _operation_map = {}

    def __init__(self, wires, *, shots=1000, analytic=True):
        super().__init__(wires, shots, analytic)
        

    def apply(self, operations, rotations=None, **kwargs):
