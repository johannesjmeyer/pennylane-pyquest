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
Custom operations
=================

**Module name:** :mod:`pennylane_forest.ops`

.. currentmodule:: pennylane_forest.ops

Sometimes the Pyquest may accept more operations
than available by core PennyLane. The plugin can define
these operations such that PennyLane can understand/apply them,
and even differentiate them.

This module contains some example PennyLane qubit operations.

The user would import them via

.. code-block:: python

    from pennylane_pyquest.ops import S, T, CCNOT

To see more details about defining custom PennyLane operations,
including more advanced cases such as defining gradient rules,
see https://pennylane.readthedocs.io/en/latest/API/overview.html

Operations
----------

.. autosummary::
    S
    T
    CCNOT
    CPHASE
    CSWAP
    ISWAP
    PSWAP


Code details
~~~~~~~~~~~~
"""
from pennylane.operation import Operation


class CompactUnitary(Operation):
    r"""CompactUnitary(alpha, beta, wires)
    CompactUnitary gate.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2 (complex)

    Args:
        alpha (complex): alpha parameter
        beta (complex): beta parameter
        wires (int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 1
    par_domain = None


class ControlledCompactUnitary(Operation):
    r"""ControlledCompactUnitary(alpha, beta, wires)
    ControlledCompactUnitary gate.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2 (complex)

    Args:
        alpha (complex): alpha parameter
        beta (complex): beta parameter
        wires (int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 2
    par_domain = None


class RotateAroundAxis(Operation):
    r"""RotateAroundAxis(theta, vector, wires)
    RotateAroundAxis gate.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2

    Args:
        theta (float): rotation angle
        vector (array[float]): rotation axis
        wires (int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 1
    par_domain = None


class RotateAroundSphericalAxis(Operation):
    r"""RotateAroundSphericalAxis(theta, spherical_theta, spherical_phi, wires)
    RotateAroundSphericalAxis gate.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3

    Args:
        theta (float): rotation angle
        spherical_theta (float): rotation axis theta
        spherical_phi (float): rotation axis phi
        wires (int): the subsystem the gate acts on
    """
    num_params = 3
    num_wires = 1
    par_domain = None


class CY(Operation):
    r"""CY(wires)
    CY gate.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None


class SqrtSWAP(Operation):
    r"""SqrtSWAP(wires)
    SqrtSWAP gate.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None


class SqrtISWAP(Operation):
    r"""SqrtISWAP(wires)
    SqrtISWAP gate.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None


class InvSqrtISWAP(Operation):
    r"""InvSqrtISWAP(wires)
    InvSqrtISWAP gate.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None


class ControlledPhaseShift(Operation):
    r"""ControlledPhaseShift(theta, wires)
    ControlledPhaseShift gate.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1

    Args:
        theta (float): rotation angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = None


class ControlledRotateAroundAxis(Operation):
    r"""ControlledRotateAroundAxis(theta, vector, wires)
    ControlledRotateAroundAxis gate.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1

    Args:
        theta (float): rotation angle
        vector (array[float]): rotation axis
        wires (int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 2
    par_domain = None


class ControlledUnitary(Operation):
    r"""ControlledUnitary(matrix, wires)
    ControlledUnitary gate.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1

    Args:
        matrix (array[complex]): the controlled unitary
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = None


class MixDephasing(Operation):
    r"""MixDephasing(probability, wires)
    MixDephasing channel.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        probability (float): dephasing probability
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = None


class MixDepolarizing(Operation):
    r"""MixDepolarizing(probability, wires)
    MixDepolarizing channel.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        probability (float): depolarization probability
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = None


class MixDamping(Operation):
    r"""MixDamping(probability, wires)
    MixDamping channel.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        probability (float): damping probability
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = None


class MixKrausMap(Operation):
    r"""MixKrausMap(kraus_operators, wires)
    MixKrausMap channel.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        kraus_operators (list[array[complex]]): Kraus operators
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = None
