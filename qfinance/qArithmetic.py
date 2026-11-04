from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from qiskit.circuit.library import QFT, MCXGate, DraperQFTAdder

def build_NMSub(m, n):
    first_number_register = QuantumRegister(n, name='phi(a)')
    second_number_register = QuantumRegister(m+1, name='b')
    quantum_circuit = QuantumCircuit(first_number_register, second_number_register, name='NMSub')
    counter = 1
    for i in range(0, m):
        for j in range(counter):
            quantum_circuit.cp(-2*np.pi/2**(counter-j), first_number_register[j], second_number_register[i])
        if counter < n:
            counter += 1
    for j in range(counter):
        quantum_circuit.cp(-2*np.pi/2**(j+2), first_number_register[n-j-1], second_number_register[-1])
    return quantum_circuit

def build_QNMSub(m, n):
    carry_register = QuantumRegister(1, name='c')
    first_number_register = QuantumRegister(n, name='a')
    second_number_register = QuantumRegister(m, name='b')
    quantum_circuit = QuantumCircuit(first_number_register, second_number_register, carry_register, name='QNMSub')
    
    quantum_circuit.ccx(first_number_register[-1], second_number_register[-1], carry_register[0], ctrl_state='10')
    quantum_circuit.ccx(first_number_register[-1], second_number_register[-1], carry_register[0], ctrl_state='01')

    qft = QFT(n+1, do_swaps=False)
    nmSub = build_NMSub(m, n).to_gate()
    inverse_qft = qft.inverse()
    quantum_circuit.append(qft.to_gate(), second_number_register[:]+carry_register[:])
    quantum_circuit.append(nmSub, first_number_register[:]+ second_number_register[:]+carry_register[:])
    quantum_circuit.append(inverse_qft.to_gate(), second_number_register[:]+carry_register[:])
    return quantum_circuit

def subtractorDraper(bitLength):
    firstRegister = QuantumRegister(bitLength, 'first')
    secondRegister = QuantumRegister(bitLength, 'second')
    carryRegister = QuantumRegister(1, 'carry')

    circuit = QuantumCircuit(firstRegister, secondRegister, carryRegister, name="subtractor")
    
    adder = DraperQFTAdder(bitLength, kind="half")
    oneIncrementer = oneIncrement(bitLength+1)
    circuit.x(secondRegister[:]+carryRegister[:])
    circuit.append(adder, firstRegister[:] + secondRegister[:] + carryRegister[:])
    circuit.append(oneIncrementer, secondRegister[:]+ carryRegister[:])
    return circuit

def oneIncrement(num_qubits):
    compute_register = QuantumRegister(num_qubits, 'compute')
    circuit = QuantumCircuit(compute_register)
    circuit.x(compute_register[0])
    circuit.cx(compute_register[0], compute_register[1], ctrl_state=0)
    for i in range(2, num_qubits):
        mcxGate = MCXGate(num_ctrl_qubits=i, ctrl_state="0"*i)
        circuit.append(mcxGate, compute_register[:i+1])
    return circuit
