# Copyright 2020 Johannes Jakob Meyer

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
from .pyquest_operation import _OPERATIONS


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
    name = "Pyquest Simulator PennyLane plugin"
    pennylane_requires = ">=0.8.0"
    version = __version__
    author = "Johannes Jakob Meyer"

    short_name = "pyquest.base"
    _operation_map = {}

    def __init__(self, wires, *, shots=1000, analytic=True):
        super().__init__(wires, shots, analytic)

    @abc.abstractmethod
    def _qureg_context(self):
        raise NotImplementedError

    def apply(self, operations, rotations=None, **kwargs):
        with self._qureg_context() as context:
            pqc.cheat.initZeroState()(qureg=context.qureg)

            for operation in operations:
                print("Applying {}".format(operation.name))
                if operation.name == "QubitStateVector":
                    pqc.cheat.initStateFromAmps()(
                        context.qureg,
                        reals=np.real(operation.parameters[0]),
                        imags=np.imag(operation.parameters[0]),
                    )
                elif operation.name == "BasisState":
                    state_int = int("".join(str(x) for x in operation.parameters[0]), 2)
                    pqc.cheat.initClassicalState()(context.qureg, state=state_int)
                else:
                    _OPERATIONS[operation.name].apply(operation, context.qureg)
