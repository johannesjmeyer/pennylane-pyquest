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

An abstract base class for constructing Target Framework devices for PennyLane.

This should contain all the boilerplate for supporting PennyLane
from the Target Framework, making it easier to create new devices.
The abstract base class below should contain all common code required
by the Target Framework.

This abstract base class will not be used by the user. Add/delete
methods and attributes below where needed.

See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of how the Device class works.

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

from pennylane import Device

from ._version import __version__


class PyquestDevice(Device):
    r"""Abstract Framework device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
        additional_option (float): as many additional arguments can be
            added as needed
    """
    name = 'Target Framework Simulator PennyLane plugin'
    pennylane_requires = '>=0.4.0'
    version = __version__
    author = 'John Smith'

    short_name = 'framework'
    _operation_map = {}

    def __init__(self, wires, *, shots=0, additional_option=2):
        super().__init__(wires, shots)
        self.additional_option = hbar
        self.prog = None
        self.state = None
        self.samples = None

    def reset(self):
        """Reset the device"""
        pass

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    # ==========================================================
    # The following methods below provide common calculations
    # and patterns required for qubit PennyLane plugin devices.
    # Feel free to keep and modify any provided methods below,
    # or delete them as you see fit.

    def expval(self, observable, wires, par):
        if self.shots == 0:
            # exact expectation value
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return (eigvals @ prob).real

        # estimate the ev
        return np.mean(self.sample(observable, wires, par))

    def var(self, observable, wires, par):
        if self.shots == 0:
            # exact variance value
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return (eigvals ** 2) @ prob - (eigvals @ prob).real ** 2

        # estimate the variance
        return np.var(self.sample(observable, wires, par))

    def rotate_basis(self, observable, wires, par):
        """Rotates the specified wires such that they
        are in the eigenbasis of the provided observable.

        Args:
            observable (str): the name of an observable
            wires (List[int]): wires the observable is measured on
            par (List[Any]): parameters of the observable
        """
        if observable == "PauliX":
            # X = H.Z.H
            self.apply("Hadamard", wires=wires, par=[])

        elif observable == "PauliY":
            # Y = (HS^)^.Z.(HS^) and S^=SZ
            self.apply("PauliZ", wires=wires, par=[])
            self.apply("S", wires=wires, par=[])
            self.apply("Hadamard", wires=wires, par=[])

        elif observable == "Hadamard":
            # H = Ry(-pi/4)^.Z.Ry(-pi/4)
            self.apply("RY", wires, [-np.pi / 4])

        elif observable == "Hermitian":
            # For arbitrary Hermitian matrix H, let U be the unitary matrix
            # that diagonalises it, and w_i be the eigenvalues.
            Hmat = par[0]
            Hkey = tuple(Hmat.flatten().tolist())

            if Hkey in self._eigs:
                # retrieve eigenvectors
                U = self._eigs[Hkey]["eigvec"]
            else:
                # store the eigenvalues corresponding to H
                # in a dictionary, so that they do not need to
                # be calculated later
                w, U = np.linalg.eigh(Hmat)
                self._eigs[Hkey] = {"eigval": w, "eigvec": U}

            # Perform a change of basis before measuring by applying U^ to the circuit
            self.apply("QubitUnitary", wires, [U.conj().T])

    def eigvals(self, observable, wires, par):
        """Determine the eigenvalues of observable(s).

        Args:
            observable (str, List[str]): the name of an observable,
                or a list of observables representing a tensor product.
            wires (List[int]): wires the observable(s) is measured on
            par (List[Any]): parameters of the observable(s)

        Returns:
            array[float]: an array of size ``(len(wires),)`` containing the
            eigenvalues of the observable
        """
        # observable should be Z^{\otimes n}
        eigvals = z_eigs(len(wires))

        if isinstance(observable, list):
            # determine the eigenvalues
            if "Hermitian" in observable:
                # observable is of the form Z^{\otimes a}\otimes H \otimes Z^{\otimes b}
                eigvals = np.array([1])

                for k, g in itertools.groupby(zip(observable, wires, par), lambda x: x[0] == "Hermitian"):
                    if k:
                        p = list(g)[0][2]
                        Hkey = tuple(p[0].flatten().tolist())
                        eigvals = np.kron(eigvals, self._eigs[Hkey]["eigval"])
                    else:
                        n = len([w for sublist in list(zip(*g))[1] for w in sublist])
                        eigvals = np.kron(eigvals, z_eigs(n))

        elif observable == "Hermitian":
            # single wire Hermitian observable
            Hkey = tuple(par[0].flatten().tolist())
            eigvals = self._eigs[Hkey]["eigval"]

        elif observable == "Identity":
            eigvals = np.ones(2 ** len(wires))

        return eigvals


@functools.lru_cache()
def z_eigs(n):
    r"""Returns the eigenvalues for :math:`Z^{\otimes n}`.

    Args:
        n (int): number of wires

    Returns:
        array[int]: eigenvalues of :math:`Z^{\otimes n}`
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([z_eigs(n - 1), -z_eigs(n - 1)])
