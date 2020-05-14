.. _usage:

Plugin usage
############

PennyLane-Plugin provides two Target Framework devices for PennyLane:

* :class:`name.pure <~PyquestPure>`: provides an PennyLane device for the Target Framework PyquestPure

* :class:`name.mixed <~PyquestMixed>`: provides an PennyLane device for the Target Framework PyquestMixed


Using the devices
=================

Once Target Framework and the plugin are installed, the two Target Framework devices
can be accessed straight away in PennyLane.

You can instantiate these devices in PennyLane as follows:

>>> import pennylane as qml
>>> from pennylane import numpy as np
>>> dev1 = qml.device('name.pure', wires=2, specific_option_for_pure=10)
>>> dev2 = qml.device('name.mixed', wires=2)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.


Device options
==============

The Target Framework simulators accept additional arguments beyond the PennyLane default device arguments.

List available device options here.

``shots=0``
	The number of circuit evaluations/random samples used to estimate expectation values of observables.
	The default value of 0 means that the exact expectation value is returned.



Supported operations
====================

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html>`_, with the exception of the PennyLane ``QubitStateVector`` state preparation operation.

In addition, the plugin provides the following framework-specific operations for PennyLane. These are all importable from :mod:`pennylane_pyquest.ops <.ops>`.

These operations include:

.. autosummary::
    pennylane_pyquest.ops.S
    pennylane_pyquest.ops.T
    pennylane_pyquest.ops.CCNOT
    pennylane_pyquest.ops.CPHASE
    pennylane_pyquest.ops.CSWAP
    pennylane_pyquest.ops.ISWAP
    pennylane_pyquest.ops.PSWAP
