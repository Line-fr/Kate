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

}

void matrixProdBench(){
    hipStream_t execthread;
    hipStreamCreate(&execthread);

    auto a = createGPUMatrix<Complex<double>>(2048);
    auto b = createGPUMatrix<Complex<double>>(2048);

    matFill(a, Complex<double>(1, 0), execthread);
    matFill(b, Complex<double>(2, 0), execthread);
    hipStreamSynchronize(execthread);
    
    hipEvent_t start, end;
    hipEventCreate(&start);
    hipEventCreate(&end);

    hipEventRecord(start, execthread);
    auto c = matProd(a, b, execthread);
    hipEventRecord(end, execthread);
    hipEventSynchronize(end);

    float time;
    hipEventElapsedTime(&time, start, end);
    cout << "the product took " << time << " ms" << endl;

    Matrix<Complex<double>> cPU = c;
    cPU.print();

    hipEventDestroy(start);
    hipEventDestroy(end);
    hipStreamDestroy(execthread);

    destroyGPUMatrix(a);
    destroyGPUMatrix(b);
    destroyGPUMatrix(c);
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

void quantumOPTTest(){
    int qbitnumber = 20;

    QuantumCircuit<double> test(qbitnumber);
    vector<int> qbits;
    Matrix<Complex<double>> mat((1llu << 3));
    mat.fill(Complex<double>(1., 0.));

    for (int i = 0; i < qbitnumber; i++){
        qbits = {i};
        test.appendGate(Gate<double>(2, qbits));
    }

    for (int i = 1; i < qbitnumber; i++){
        qbits = {0, i};
        test.appendGate(Gate<double>(3, qbits));
    }
    for (int i = 1; i < qbitnumber; i++){
        qbits = {0, i};
        test.appendGate(Gate<double>(3, qbits));
    }

    test.print();

    test.gateScheduling();

    cout << endl;
    cout << "scheduling done" << endl;
    test.print();

    test.gateFusion(6, 0.01);

    cout << endl;
    cout << "merge done!" << endl;
    test.print();
    //cout << "value at 00 of the last gate is : ";
    //test.gate_set_ordered[test.gate_set_ordered.size()-1].densecontent(0, 0).print();
    cout << endl;

    test.gateGrouping(8);
    cout << endl;
    cout << "groups done!" << endl;
    test.print();

    test.allocate(27, 0);
    cout << endl;
    cout << "allocate done!" << endl;
    test.print();

    Simulator<double>(test, 1);

    cout << endl;
}

int main(){
    printGpuInfo<Complex<double>>();

    //quantumOPTTest();

    int qbitnumber = 25;
    int gpulog2 = 0;

    QuantumCircuit<double> test(qbitnumber);
    vector<int> qbits;
    Matrix<Complex<double>> mat((1llu << 5));
    mat.fill(Complex<double>(1., 0.));

    for (int i = 0; i < qbitnumber; i+= 5){
        qbits = {i, i+1, i+2, i+3, i+4};
        test.appendGate(Gate<double>(mat, qbits));
    }

    int l = 0;
    vector<int> shuffled = {24, 18, 25, 26, 27, 28, 29, 30, 9, 6, 5, 17, 21, 16, 11, 0, 3, 22, 15, 23, 8, 1, 10, 20, 7, 19, 14, 12, 2, 4, 13};
    for (const auto& i: shuffled){
        if (i >= qbitnumber) continue;
        qbits = {l};
        test.appendGate(Gate<double>(2, qbits));
        l++;
    }

    test.gateScheduling();
    test.gateFusion();
    test.gateGrouping();
    test.allocate(28, gpulog2);

    test.print();

    Simulator<double> sim(test, (1 << gpulog2));

    sim.execute(true);
    
    cout << "done" << endl;

    //todo: implement measurment, implement QFT circuit with new gates
    //better gate scheduling euristic!
    //lastly, details optimizations

    return 0;
}