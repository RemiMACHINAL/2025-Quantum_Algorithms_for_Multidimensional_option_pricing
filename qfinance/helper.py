from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
        
def loadNumber(circ: QuantumCircuit, register: QuantumRegister, number: int):
    number_in_binary = bin(number)[2:].zfill(register.size)
    for i in range(len(number_in_binary)):
        if number_in_binary[::-1][i] == "1":
            circ.x(register[i])
    return circ


