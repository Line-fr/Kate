#include<iostream>
#include<hip/hip_runtime.h>
//#include "matrix.hpp"
//#include "gate.hpp"
#include "simulator.hpp"
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<chrono>
#define THREADSNUM 256

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using namespace std;

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

__global__ void testkernel(){
    printf("gpu kernel works\n");
}

template<typename T>
void printGpuInfo(){
    int count, device;
    hipDeviceProp_t devattr;
	if (hipGetDeviceCount(&count) != 0){
		cout << "couldnt detect devices, check permissions" << endl;
		return;
	}
    for (int i = 0; i < count; i++){
        GPU_CHECK(hipSetDevice(i));
        GPU_CHECK(hipGetDevice(&device));
	    GPU_CHECK(hipGetDeviceProperties(&devattr, device));
        cout << "GPU " << i << " : " << devattr.name << endl;
    }
    cout << "-----------------------" << endl;
    GPU_CHECK(hipSetDevice(0));
	GPU_CHECK(hipGetDevice(&device));
	GPU_CHECK(hipGetDeviceProperties(&devattr, device));
	cout << "current GPU: " << endl;
	cout << devattr.name << endl;
    cout << endl;
    cout << "Global memory: " << devattr.totalGlobalMem << " (" << (int)log2(devattr.totalGlobalMem/sizeof(T)) << " qbits)" << endl;
    cout << "Shared memory per block : " << devattr.sharedMemPerBlock << " (" << (int)log2(devattr.sharedMemPerBlock/sizeof(T))/2 << " max qbits dense gate execution)" << endl;
    cout << "Registers per blocks : " << devattr.regsPerBlock << " (" << (int)log2(devattr.regsPerBlock/THREADSNUM) << " max qbits group)" << endl;
	cout << endl;
    testkernel<<<dim3(1), dim3(1)>>>();
    GPU_CHECK(hipDeviceSynchronize());
}

int main(){
    printGpuInfo<Complex<double>>();

    //quantumOPTTest();

    int nqbits = 28;

    QuantumCircuit<double> circuit(nqbits);
    vector<int> qbits;

    for (int i = 1; i < nqbits; i++){
        qbits = {i};
        circuit.appendGate(Gate<double>(2, qbits));
    }

    circuit.print();

    circuit.compileOPT(5, 0.00001, 10, 0, 28);

    circuit.print();

    Simulator<double> sim(circuit, 1);

    proba_state res = sim.execute(true);

    res.print();

    res = sim.execute(res, true);

    res.print();

    /*
    auto t1 = high_resolution_clock::now();
    auto t2 = high_resolution_clock::now();
    auto circuit = randomCircuit<double>(25, 10);
    double sum = 0;
    for (int i = 0; i < 100; i++){
        circuit = randomCircuit<double>(25, 10);
        t1 = high_resolution_clock::now();
        circuit.compileOPT(5, 1, 10, 0);
        Simulator<double>(circuit, 1).execute(true);
        t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double_compute = t2 - t1;
        sum += ms_double_compute.count();
    }
    cout << "average time : " << sum/100 << " ms" << endl;*/

    //todo: implement measurment, implement QFT circuit with new gates
    //better gate scheduling euristic!
    //lastly, details optimizations

    cout << "done" << endl;

    return 0;
}