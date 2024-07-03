#ifndef BENCHCIRCUITSDONE
#define BENCHCIRCUITSDONE

#include "Kate.hpp"

namespace Kate{

Circuit QulacsBench(int qbitsNum){
    srand (time(NULL));
    Kate::Circuit circuit(qbitsNum);
    std::vector<int> qbits;
    for (int i = 0; i < qbitsNum; i++){
        qbits = {i};
        circuit.appendGate(Gate(Hadamard, qbits, 0));
        circuit.appendGate(Gate(Hadamard, qbits, 0));
    }
    for (int i = 0; i < qbitsNum; i++){
        qbits = {i, (i+1)%qbitsNum};
        circuit.appendGate(Gate(CNOT, qbits));
    }

    for (int a = 0; a < 9; a++){
        for (int i = 0; i < qbitsNum; i++){
            qbits = {i};
            circuit.appendGate(Gate(Hadamard, qbits, 0));
            circuit.appendGate(Gate(Hadamard, qbits, 0));
            circuit.appendGate(Gate(Hadamard, qbits, 0));
        }
        for (int i = 0; i < qbitsNum; i++){
            qbits = {i, (i+1)%qbitsNum};
            circuit.appendGate(Gate(CNOT, qbits));
        }
    }

    for (int i = 0; i < qbitsNum; i++){
        qbits = {i};
        circuit.appendGate(Gate(Hadamard, qbits, 0));
        circuit.appendGate(Gate(Hadamard, qbits, 0));
    }

    return circuit;
}

Circuit QFT(int qbitsNum){
    Circuit res(qbitsNum);
    std::vector<int> qbits;

    for (int i = 0; i < qbitsNum; i++){
        qbits = {i};
        res.appendGate(Gate(Hadamard, qbits));
        for (int j = i+1; j < qbitsNum; j++){
            qbits = {j, i}; //first element is controller which here is j
            res.appendGate(Gate(CRk, qbits, j - i +1));
        }
    }

    return res;
}

Circuit randomCircuit(int qbitsNum, int depth){
    srand (time(NULL));
    Circuit res(qbitsNum);
    std::vector<int> qbits;
    Matrix<Complex> mat2((1llu << (2)));
    Matrix<Complex> mat3((1llu << (3)));
    mat2.fill(1.); //the content do not matter for benchmarking
    mat3.fill(1.);

    int temp;
    for (int i = 0; i < depth; i++){
        switch(rand()%3){
            case 0: //1 qbit gate, we will put H because it is well suited for benchmarcking (dense)
                temp = rand()%qbitsNum;
                qbits = {temp};
                res.appendGate(Gate(Hadamard, qbits));
                break;
            case 1: //2 qbit gate, either CNOT (sparse) or mat2 (dense)
                temp = rand()%qbitsNum;
                qbits = {temp};
                while (temp == qbits[0]){ //yeah... I know
                    temp = rand()%qbitsNum;
                }
                qbits.push_back(temp);
                if (rand()%2 == 1){
                    res.appendGate(Gate(CNOT, qbits));
                } else {
                    res.appendGate(Gate(mat2, qbits));
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
                    res.appendGate(Gate(TOFFOLI, qbits));
                } else {
                    res.appendGate(Gate(mat3, qbits));
                }
                break;
        }
    }
    return res;
}

}

#endif