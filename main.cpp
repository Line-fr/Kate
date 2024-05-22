#include<iostream>
#include<hip/hip_runtime.h>
//#include "matrix.hpp"
//#include "gate.hpp"
#include "simulator.hpp"
#include<math.h>
#define THREADSNUM 256

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
        hipSetDevice(i);
        hipGetDevice(&device);
	    hipGetDeviceProperties(&devattr, device);
        cout << "GPU " << i << " : " << devattr.name << endl;
    }
    cout << "-----------------------" << endl;
    hipSetDevice(0);
	hipGetDevice(&device);
	hipGetDeviceProperties(&devattr, device);
	cout << "current GPU: " << endl;
	cout << devattr.name << endl;
    cout << endl;
    cout << "Global memory: " << devattr.totalGlobalMem << " (" << (int)log2(devattr.totalGlobalMem/sizeof(T)) << " qbits)" << endl;
    cout << "Shared memory per block : " << devattr.sharedMemPerBlock << " (" << (int)log2(devattr.sharedMemPerBlock/sizeof(T))/2 << " max qbits dense gate execution)" << endl;
    cout << "Registers per blocks : " << devattr.regsPerBlock << " (" << (int)log2(devattr.regsPerBlock/THREADSNUM) << " max qbits group)" << endl;
	cout << endl;
    testkernel<<<dim3(1), dim3(1)>>>();
    hipDeviceSynchronize();
}

int main(){
    printGpuInfo<Complex<double>>();

    //quantumOPTTest();

    auto circuit = QFT<double>(24);

    circuit.print();

    circuit.compileOPT(5, 0.00001, 10, 0);

    circuit.print();

    Simulator<double> sim(circuit, 1);

    sim.execute(true);

    //todo: implement measurment, implement QFT circuit with new gates
    //better gate scheduling euristic!
    //lastly, details optimizations

    cout << "done" << endl;

    return 0;
}