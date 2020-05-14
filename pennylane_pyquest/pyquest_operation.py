# Copyright 2020 Johannes Jakob Meyer Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pyquest_cffi as pqc
import pennylane as qml


class PyquestOperation:
    def __init__(self, converter):
        # Takes a PL operation and makes a function that applies said function to a qreg
        self.converter = converter

    def apply(self, operation, qureg):
        self.converter(operation)(qureg)


_OPERATIONS = {
    "Hadamard": PyquestOperation(
        lambda op, qureg: pqc.ops.hadamard()(qureg=qureg, qubit=op.wires[0])
    ),
    "PauliX": PyquestOperation(
        lambda op, qureg: pqc.ops.pauliX()(qureg=qureg, qubit=op.wires[0])
    ),
    "PauliY": PyquestOperation(
        lambda op, qureg: pqc.ops.pauliY()(qureg=qureg, qubit=op.wires[0])
    ),
    "PauliZ": PyquestOperation(
        lambda op, qureg: pqc.ops.pauliZ()(qureg=qureg, qubit=op.wires[0])
    ),
    "S": PyquestOperation(
        lambda op, qureg: pqc.ops.sGate()(qureg=qureg, qubit=op.wires[0])
    ),
    "T": PyquestOperation(
        lambda op, qureg: pqc.ops.tGate()(qureg=qureg, qubit=op.wires[0])
    ),
    "CompactUnitary": PyquestOperation(
        lambda op, qureg: pqc.ops.compactUnitary()(
            qureg=qureg,
            qubit=op.wires[0],
            alpha=op.parameters[0],
            beta=op.parameters[1],
        )
    ),  # Custom
    "PhaseShift": PyquestOperation(
        lambda op, qureg: pqc.ops.phaseShift()(
            qureg=qureg, qubit=op.wires[0], theta=op.parameters[0]
        )
    ),
    "RotateAroundAxis": PyquestOperation(
        lambda op, qureg: pqc.ops.rotateAroundAxis()(
            qureg=qureg,
            qubit=op.wires[0],
            theta=op.parameters[0],
            vector=op.parameters[1],
        )
    ),  # Custom
    "RotateAroundSphericalAxis": PyquestOperation(
        lambda op, qureg: pqc.ops.rotateAroundAxis()(
            qureg=qureg,
            qubit=op.wires[0],
            theta=op.parameters[0],
            spherical_theta=op.parameters[1],
            spherical_phi=[2],
        )
    ),  # Custom
    "RX": PyquestOperation(
        lambda op, qureg: pqc.ops.rotateX()(
            qureg=qureg, qubit=op.wires[0], theta=op.parameters[0]
        )
    ),
    "RY": PyquestOperation(
        lambda op, qureg: pqc.ops.rotateY()(
            qureg=qureg, qubit=op.wires[0], theta=op.parameters[0]
        )
    ),
    "RZ": PyquestOperation(
        lambda op, qureg: pqc.ops.rotateZ()(
            qureg=qureg, qubit=op.wires[0], theta=op.parameters[0]
        )
    ),
    "QubitUnitary": PyquestOperation(
        lambda op, qureg: pqc.ops.unitary()(
            qureg=qureg, qubit=op.wires[0], matrix=op.parameters[0]
        )
    ),  # Only single qubit
    "ControlledCompactUnitary": PyquestOperation(
        lambda op, qureg: pqc.ops.compactUnitary()(
            qureg=qureg,
            control=op.wires[0],
            qubit=op.wires[1],
            alpha=op.parameters[0],
            beta=op.parameters[1],
        )
    ),  # Custom
}
