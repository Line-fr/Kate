#ifndef BENCHCIRCUITSDONE
#define BENCHCIRCUITSDONE

#include "preprocessor.hpp"
#include "QuantumCircuit.hpp"

QuantumCircuit<double> QulacsBench(int qbitsNum){
    srand (time(NULL));
    QuantumCircuit<double> circuit(qbitsNum);
    vector<int> qbits;
    for (int i = 0; i < qbitsNum; i++){
        qbits = {i};
        circuit.appendGate(Gate<double>(Hadamard, qbits, 0));
        circuit.appendGate(Gate<double>(Hadamard, qbits, 0));
    }
    for (int i = 0; i < qbitsNum; i++){
        qbits = {i, (i+1)%qbitsNum};
        circuit.appendGate(Gate<double>(CNOT, qbits));
    }

    for (int a = 0; a < 9; a++){
        for (int i = 0; i < qbitsNum; i++){
            qbits = {i};
            circuit.appendGate(Gate<double>(Hadamard, qbits, 0));
            circuit.appendGate(Gate<double>(Hadamard, qbits, 0));
            circuit.appendGate(Gate<double>(Hadamard, qbits, 0));
        }
        for (int i = 0; i < qbitsNum; i++){
            qbits = {i, (i+1)%qbitsNum};
            circuit.appendGate(Gate<double>(CNOT, qbits));
        }
    }

    for (int i = 0; i < qbitsNum; i++){
        qbits = {i};
        circuit.appendGate(Gate<double>(Hadamard, qbits, 0));
        circuit.appendGate(Gate<double>(Hadamard, qbits, 0));
    }

    return circuit;
}

template<typename T>
QuantumCircuit<T> QFT(int qbitsNum){
    QuantumCircuit<T> res(qbitsNum);
    vector<int> qbits;

    for (int i = 0; i < qbitsNum; i++){
        qbits = {i};
        res.appendGate(Gate<T>(Hadamard, qbits));
        for (int j = i+1; j < qbitsNum; j++){
            qbits = {j, i}; //first element is controller which here is j
            res.appendGate(Gate<T>(CRk, qbits, j - i +1));
        }
    }

    return res;
}

template<typename T>
QuantumCircuit<T> randomCircuit(int qbitsNum, int depth){
    srand (time(NULL));
    QuantumCircuit<T> res(qbitsNum);
    vector<int> qbits;
    Matrix<Complex<T>> mat2((1llu << (2)));
    Matrix<Complex<T>> mat3((1llu << (3)));
    mat2.fill(1.); //the content do not matter for benchmarking
    mat3.fill(1.);

    int temp;
    for (int i = 0; i < depth; i++){
        switch(rand()%3){
            case 0: //1 qbit gate, we will put H because it is well suited for benchmarcking (dense)
                temp = rand()%qbitsNum;
                qbits = {temp};
                res.appendGate(Gate<T>(Hadamard, qbits));
                break;
            case 1: //2 qbit gate, either CNOT (sparse) or mat2 (dense)
                temp = rand()%qbitsNum;
                qbits = {temp};
                while (temp == qbits[0]){ //yeah... I know
                    temp = rand()%qbitsNum;
                }
                qbits.push_back(temp);
                if (rand()%2 == 1){
                    res.appendGate(Gate<T>(CNOT, qbits));
                } else {
                    res.appendGate(Gate<T>(mat2, qbits));
                }
                break;
            case 2: //3 qbit gate, either toffoli (sparse) or mat3 (dense)
                temp = rand()%qbitsNum;
                qbits = {temp};
                while (temp == qbits[0]){ //yeah... I know
                    temp = rand()%qbitsNum;
                }
                qbits.push_back(temp);
                while (temp == qbits[0] || temp == qbits[1]){ //yeah... I know
                    temp = rand()%qbitsNum;
                }
                qbits.push_back(temp);
                if (rand()%2 == 1){
                    res.appendGate(Gate<T>(TOFFOLI, qbits));
                } else {
                    res.appendGate(Gate<T>(mat3, qbits));
                }
                break;
        }
    }
    return res;
}

#endif