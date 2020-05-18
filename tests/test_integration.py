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
"""Tests that plugin devices are accessible and integrate with PennyLane"""
import numpy as np
import pennylane as qml
import pytest
from pennylane_pyquest import PyquestPure, PyquestMixed
import autograd

from conftest import shortnames


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("d", shortnames)
    def no_test_load_device(self, d):
        """Test that the QVM device loads correctly"""
        dev = qml.device(d, wires=2, shots=1024)
        assert dev.num_wires == 2
        assert dev.shots == 1024
        assert dev.short_name == d

    @pytest.mark.parametrize("d", shortnames)
    @pytest.mark.parametrize("shots", [1000, 8192])
    def test_one_qubit_circuit(self, shots, d, tol):
        """Test that devices provide correct result for a simple circuit"""
        _dev = PyquestPure if d == "pyquest.pure" else PyquestMixed
        dev = _dev(wires=1, shots=shots)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

@pytest.mark.parametrize("shots", [8192])
class TestCompareDefaultQubit:
    """Integration tests against default.qubit"""

    @pytest.mark.parametrize("params", np.random.uniform(0, 2*np.pi, (10, 3, 3, 3)))
    def test_strongly_ent_layers(self, device, shots, tol, params):
        dev = device(3)
        comp_dev = qml.device("default.qubit", wires=3)

        def circuit(params):
            qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2])

            return qml.expval(qml.PauliX(0)), qml.var(qml.Hadamard(1)), qml.expval(qml.Hermitian(np.array([[1, 2], [2, -4]]), wires=[2]))

        node = qml.QNode(circuit, dev)
        comp_node = qml.QNode(circuit, comp_dev)

        assert np.allclose(node(params), comp_node(params), **tol)

    @pytest.mark.parametrize("params", np.random.uniform(0, 2*np.pi, (10, 14)))
    def test_arbitrary_state_prep(self, device, shots, tol, params):
        dev = device(3)
        comp_dev = qml.device("default.qubit", wires=3)

        def circuit(params):
            qml.templates.ArbitraryStatePreparation(params, wires=[0, 1, 2])

            return qml.expval(qml.PauliX(0)), qml.var(qml.Hadamard(1)), qml.expval(qml.Hermitian(np.array([[1, 2], [2, -4]]), wires=[2]))

        node = qml.QNode(circuit, dev)
        comp_node = qml.QNode(circuit, comp_dev)

        assert np.allclose(node(params), comp_node(params), **tol)

    @pytest.mark.parametrize("params", np.random.uniform(0, 2*np.pi, (3, 14)))
    def test_diff(self, device, shots, tol, params):
        dev = device(3)
        comp_dev = qml.device("default.qubit", wires=3)

        def circuit(params):
            qml.templates.ArbitraryStatePreparation(params, wires=[0, 1, 2])

            return qml.expval(qml.PauliX(0)), qml.var(qml.Hadamard(1)), qml.expval(qml.Hermitian(np.array([[1, 2], [2, -4]]), wires=[2]))

        node = qml.QNode(circuit, dev)
        comp_node = qml.QNode(circuit, comp_dev)

        cost = lambda params: autograd.numpy.sum(node(params)) - node(params)[0]**2
        comp_cost = lambda params: autograd.numpy.sum(comp_node(params)) - comp_node(params)[0]**2

        print("cost(params) = ", cost(params))
        print("comp_cost(params) = ", comp_cost(params))

        assert np.allclose(autograd.grad(cost)(params), autograd.grad(comp_cost)(params), **tol)
