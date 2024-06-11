#include<hip/hip_runtime.h>
#include "matrix.hpp"
#include<vector>
#include<set>
#include<math.h>
#include "graphUtil.hpp"
#include <limits.h>
#include<algorithm>
#include<list>
#include<chrono>

#define PI 3.1415926535897932384626433
#define SQRT2 1.4142135623730951
#define SQRT2INV 0.7071067811865475
#define Hadamard 2
#define CNOT 3
#define CRk 4
#define TOFFOLI 5
#define RX 6
#define RZ 7

#define GPU_CHECK(x)\
err = (x);\
if (err != hipSuccess)\
{\
   	cout << hipGetErrorString(x) << " in " << __FILE__ << " at line " << __LINE__ << endl;\
}

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

template<typename T>
__device__ Complex<T> exp(Complex<T> i){
    double temp = exp(i.a);
    return Complex<T>(temp*cos(i.b), temp*sin(i.b));
}

set<int> union_elements(set<int>& a, set<int>& b){
    set<int> c = b;
    for (const auto& el: a){
        c.insert(el);
    }
    return c;
}

set<int> union_elements(set<int>& a, vector<int>& b){
    set<int> c = a;
    for (const auto& el: b){
        c.insert(el);
    }
    return c;
}

template<typename T>
class GPUGate;

template<typename T>
class Gate{
public:
    int identifier = -1; //0 is dense, 2 is H, 3 is CNOT,...
    int optarg;
    double optarg2;
    Complex<T> optarg3;
    Matrix<Complex<T>> densecontent = NULL;
    vector<int> qbits;
    Gate(int identifier, vector<int>& qbits, int optarg = 0, double optarg2 = 0, Complex<T> optarg3 = 0){
        this->identifier = identifier;
        this->optarg = optarg;
        this->optarg2 = optarg2;
        this->optarg3 = optarg3;
        if (identifier == 0) {
            cout << "you are creating a dense matrix without specifying it. Memory error incoming" << endl;
        }
        if (identifier == 2 && qbits.size() != 1){
            cout << "hadamard is on exactly 1 qbit" << endl;
        }
        if (identifier == 3 && qbits.size() != 2){
            cout << "CNOT is on exactly 2 qbits" << endl;
        }
        if (identifier == 4 && qbits.size() != 2){
            cout << "Controlled Rk is on exactly 2 qbits" << endl;
        }
        if (identifier == 5 && qbits.size() != 3){
            cout << "toffoli is on exactly 3 qbits" << endl;
        }
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
    }
    Gate(Matrix<Complex<T>>& densecontent, vector<int>& qbits){
        identifier = 0;
        this->densecontent = densecontent;
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
        if ((1llu << qbits.size()) != densecontent.n){
            cout << "size mismatch dense matrix dimension error" << endl;
        }
    }
    Gate(int identifier, Matrix<Complex<T>>& densecontent, vector<int>& qbits){
        this->identifier = identifier;
        this->densecontent = densecontent;
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
        if ((1llu << qbits.size()) != densecontent.n){
            cout << "size mismatch dense matrix dimension error" << endl;
        }
    }
    Gate(const GPUGate<T>& other){
        identifier = other.identifier;
        densecontent = Matrix<Complex<T>>(other.densecontent);
        qbits.clear();
        int* temp = (int*)malloc(sizeof(int)*other.nbqbits);
        GPU_CHECK(hipMemcpyDtoH(temp, (hipDeviceptr_t)other.qbits, sizeof(int)*other.nbqbits));
        for (int i = 0; i < other.nbqbits; i++){
            qbits.push_back(temp[i]);
        }
        free(temp);
    }
    void print(){
        cout << "Identifier : " << identifier << ", Qbits affected : ";
        for (const auto& el: qbits){
            cout << el << " ";
        }
        cout << endl;
    }
};

template<typename T>
class GPUGate{
public:
    int identifier = -1;
    int optarg;
    double optarg2;
    Complex<T> optarg3;
    GPUMatrix<Complex<T>> densecontent;
    int* qbits = NULL;
    int* ordered_qbits = NULL;
    int nbqbits = 0;
    //gates[gateid], nqbits, blockDim.x, threadIdx.x, blockIdx.x, qbitsstate, sharedMemMatrixSize, matrixsharedstorage, bit_to_groupbitnumber
    __device__ void compute(int nqbits, Complex<T>* qbitsstateshared, int* bit_to_groupbitnumber, Complex<T>* matrixsharedstorage, int sharedMemMatrixSize){
        size_t initline = threadIdx.x;
        size_t beg, end;
        size_t begmat, endmat;
        switch(identifier){
            case TOFFOLI: {
                size_t to_cover = (1llu << (nqbits - 3));
                size_t work_per_thread  = to_cover/blockDim.x;
                if (work_per_thread == 0){ //not enough qbits to fully utilize even a simple block... consider putting less threads per block or using cpu here
                    beg = initline;
                    end = (initline < to_cover) ? initline+1 : initline;
                } else {
                    beg = initline*work_per_thread;
                    end = (initline+1)*work_per_thread;
                }
                //we don't even need to put the gate in memory since it s not dense, let's get our indexes
                int lq0 = bit_to_groupbitnumber[qbits[0]];
                int lq1 = bit_to_groupbitnumber[qbits[1]];
                int lq2 = bit_to_groupbitnumber[qbits[2]];
                
                size_t mask0, mask1, mask2, mask3;
                mask0 = (1llu << (bit_to_groupbitnumber[ordered_qbits[0]])) - 1;
                mask1 = (1llu << (bit_to_groupbitnumber[ordered_qbits[1]] - 1)) - 1 - mask0;
                mask2 = (1llu << (bit_to_groupbitnumber[ordered_qbits[2]] - 2)) - 1 - mask0 - mask1;
                mask3 = (1llu << (nqbits-3)) - 1 - mask0 - mask1 - mask2;
                for (size_t line = beg; line < end; line++){
                    size_t index110 = (1llu << lq0) + (1llu << lq1) + (line&mask0) + ((line&mask1) << (1)) + ((line&mask2) << (2)) + ((line&mask3) << (3)); //XXXXX-lq1(1)-XXXXX-lq0(0)-XXXXX
                    size_t index111 = index110 + (1llu << lq2);
                    auto temp = qbitsstateshared[index110];
                    qbitsstateshared[index110] = qbitsstateshared[index111];
                    qbitsstateshared[index111] = temp;
                }
                break;
            }
            case CRk: {
                size_t to_cover = (1llu << (nqbits - 2));
                size_t work_per_thread  = to_cover/blockDim.x;
                if (work_per_thread == 0){ //not enough qbits to fully utilize even a simple block... consider putting less threads per block or using cpu here
                    beg = initline;
                    end = (initline < to_cover) ? initline+1 : initline;
                } else {
                    beg = initline*work_per_thread;
                    end = (initline+1)*work_per_thread;
                }
                //we don't even need to put the gate in memory since it s not dense, let's get our indexes
                int lq0 = bit_to_groupbitnumber[qbits[0]];
                int lq1 = bit_to_groupbitnumber[qbits[1]];
                size_t mask0, mask1, mask2;
                mask0 = (1llu << (bit_to_groupbitnumber[ordered_qbits[0]])) - 1;
                mask1 = (1llu << (bit_to_groupbitnumber[ordered_qbits[1]] - 1)) - 1 - mask0;
                mask2 = (1llu << (nqbits-2)) - 1 - mask0 - mask1;
                for (size_t line = beg; line < end; line++){
                    size_t index10 = (1llu << lq1) + (line&mask0) + ((line&mask1) << (1)) + ((line&mask2) << (2)); //XXXXX-lq1(1)-XXXXX-lq0(0)-XXXXX
                    size_t index11 = index10 + (1llu << lq0);
                    double temp = ((double)2*PI)/(1llu << (optarg));
                    qbitsstateshared[index11] *= Complex<T>(cos(temp), sin(temp));
                }
                break;
            }
            case CNOT: {//CNOT
                //CNOT being a small gate, it is more interesting to make parallel the index of qbitstates
                size_t to_cover = (1llu << (nqbits - 2));
                size_t work_per_thread  = to_cover/blockDim.x;
                if (work_per_thread == 0){ //not enough qbits to fully utilize even a simple block... consider putting less threads per block or using cpu here
                    beg = initline;
                    end = (initline < to_cover) ? initline+1 : initline;
                } else {
                    beg = initline*work_per_thread;
                    end = (initline+1)*work_per_thread;
                }
                //we don't even need to put the gate in memory since it s not dense, let's get our indexes
                int lq0 = bit_to_groupbitnumber[qbits[0]];
                int lq1 = bit_to_groupbitnumber[qbits[1]];
                size_t mask0, mask1, mask2;
                mask0 = (1llu << (bit_to_groupbitnumber[ordered_qbits[0]])) - 1;
                mask1 = (1llu << (bit_to_groupbitnumber[ordered_qbits[1]] - 1)) - 1 - mask0;
                mask2 = (1llu << (nqbits-2)) - 1 - mask0 - mask1;
                for (size_t line = beg; line < end; line++){
                    size_t index10 = (1llu << lq1) + (line&mask0) + ((line&mask1) << (1)) + ((line&mask2) << (2)); //XXXXX-lq1(1)-XXXXX-lq0(0)-XXXXX
                    size_t index11 = index10 + (1llu << lq0);
                    auto temp = qbitsstateshared[index10];
                    qbitsstateshared[index10] = qbitsstateshared[index11];
                    qbitsstateshared[index11] = temp;
                }
                break;
            }
            case Hadamard: { //H
                //H being a small gate, it is more interesting to make parallel the index of qbitstates
                size_t to_cover = (1llu << (nqbits - 1));
                size_t work_per_thread  = to_cover/blockDim.x;
                if (work_per_thread == 0){ //not enough qbits to fully utilize even a simple block... consider putting less threads per block or using cpu here
                    beg = initline;
                    end = (initline < to_cover) ? initline+1 : initline;
                } else {
                    beg = initline*work_per_thread;
                    end = (initline+1)*work_per_thread;
                }
                //we don't even need to put the gate in memory since it s not dense, let's get our indexes
                int lq0 = bit_to_groupbitnumber[qbits[0]];
                size_t mask0, mask1;
                mask0 = (1llu << (lq0)) - 1;
                mask1 = (1llu << (nqbits - 1)) - 1 - mask0;
                for (size_t line = beg; line < end; line++){
                    size_t index0 = (line&mask0) + ((line&mask1) << (1)); //XXXXX-lq0(0)-XXXXX
                    size_t index1 = index0 + (1llu << lq0);
                    //printf("index 0 : %llu , index 1 : %llu for lq0 : %i\n values before execution : %f, %f\n", index0, index1, lq0, qbitsstateshared[index0].a, qbitsstateshared[index1].a);
                    qbitsstateshared[index0] = (qbitsstateshared[index0]+qbitsstateshared[index1])*SQRT2INV;
                    qbitsstateshared[index1] = qbitsstateshared[index0] - qbitsstateshared[index1]*SQRT2;
                    //printf("index 0 : %llu , index 1 : %llu for lq0 : %i\n values after execution : %f, %f\n", index0, index1, lq0, qbitsstateshared[index0].a, qbitsstateshared[index1].a);
                }
                break;
            }
            case 0: { //Dense
                //DENSE has 2 types of parallelization: matrix product and lines. ideally, we would use both of them to account for all matrix size cases.
                //for that purpose, let's parrallelize lines as much as possible, and allocate if there are some for the matrix product
                int gateqbits = nbqbits;
                size_t to_cover = (1llu << (nqbits - gateqbits));
                size_t work_per_thread  = to_cover/blockDim.x;
                if (work_per_thread == 0){ //lines was not able to utilize all threads so let's dive into matrix product here
                    size_t remaining_threads_per_mat = blockDim.x/to_cover;
                    beg = initline/remaining_threads_per_mat;
                    end = beg+1;
                    begmat = (initline%remaining_threads_per_mat)*(1llu << gateqbits)/remaining_threads_per_mat;
                    endmat = begmat + (1llu << gateqbits)/remaining_threads_per_mat;
                } else {
                    beg = initline*work_per_thread;
                    end = beg+work_per_thread;
                    begmat = 0;
                    endmat = (1llu << gateqbits);
                }
                //let's see if we can put this matrix in shared memory!
                Complex<T>* matrixdata;
                if ((sharedMemMatrixSize >= sizeof(Complex<T>)*(1llu << (2*gateqbits)))){
                    if (blockDim.x > (1llu << (2*gateqbits))){
                        if (initline < (1llu << (2*gateqbits))) matrixsharedstorage[initline] = densecontent.data[initline];
                    } else {
                        size_t ratio = (1llu << (2*gateqbits))/blockDim.x;
                        for (int i = initline*ratio; i < (initline+1)*ratio; i++){
                            matrixsharedstorage[i] = densecontent.data[i];
                        }
                    }
                    matrixdata = matrixsharedstorage;
                    __syncthreads();
                } else {
                    matrixdata = densecontent.data;
                }
                //now it's time to build masks
                size_t masks[64]; //I want them in registers so better invoke them in a fixed size array. will work for as much as 63 qbits.
                size_t cumulative = 0;
                for (int i = 0; i < gateqbits; i++){
                    masks[i] = (1llu << (bit_to_groupbitnumber[ordered_qbits[i]] - i)) - 1 - cumulative;
                    cumulative += masks[i];
                }
                masks[gateqbits] = (1llu << (nqbits - gateqbits)) - 1 - cumulative;

                for (size_t line = beg; line < end; line++){
                    size_t baseind = 0; //will be XXXXX-0-XXXX-0-XXXXXX-0-...XXXX;
                    for (int i = 0; i <= gateqbits; i++){
                        baseind += ((line&masks[i]) << i);
                    }
                    for (size_t matline = begmat; matline < endmat; matline++){
                        size_t tempind;
                        size_t lineind = baseind;
                        for (int i = 0; i < gateqbits; i++){
                            lineind += ((matline >> i)%2) << bit_to_groupbitnumber[qbits[i]];
                        }
                        Complex<T> sum = 0;
                        for (size_t matcol = 0; matcol < (1llu << gateqbits); matcol++){

                            tempind = baseind;
                            for (int i = 0; i < gateqbits; i++){
                                tempind += ((matcol >> i)%2) << bit_to_groupbitnumber[qbits[i]];
                            }
                            sum += matrixdata[(matline << gateqbits) + matcol]*qbitsstateshared[tempind];
                        }
                        qbitsstateshared[lineind] = sum;
                        
                    }
                }
                break;
            }
        }
    }
};

template<typename T>
GPUGate<T> createGPUGate(const Gate<T>& other){
    GPUGate<T> res;
    res.identifier = other.identifier;
    res.optarg = other.optarg;
    res.optarg2 = other.optarg2;
    res.optarg3 = other.optarg3;
    res.densecontent = createGPUMatrix<Complex<T>>(other.densecontent);
    res.nbqbits = other.qbits.size();
    GPU_CHECK(hipMalloc(&res.qbits, sizeof(int)*res.nbqbits));
    GPU_CHECK(hipMalloc(&res.ordered_qbits, sizeof(int)*res.nbqbits));
    int i = 0;
    int* temp = (int*)malloc(sizeof(int)*res.nbqbits);
    for (const auto& el: other.qbits){
        temp[i] = el;
        i++;
    }
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)res.qbits, temp, sizeof(int)*res.nbqbits));
    i = 0;
    for (const auto& el: set<int>(other.qbits.begin(), other.qbits.end())){
        temp[i] = el;
        i++;
    }
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)res.ordered_qbits, temp, sizeof(int)*res.nbqbits));
    free(temp);
    return res;
}

void freecallback(hipStream_t stream, hipError_t err, void* data){
    free(data);
}

template<typename T>
GPUGate<T> createGPUGateAsync(const Gate<T>& other){
    GPUGate<T> res;
    res.identifier = other.identifier;
    res.optarg = other.optarg;
    res.densecontent = createGPUMatrixAsync<Complex<T>>(other.densecontent);
    res.nbqbits = other.qbits.size();
    GPU_CHECK(hipMalloc(&res.qbits, sizeof(int)*res.nbqbits));
    GPU_CHECK(hipMalloc(&res.ordered_qbits, sizeof(int)*res.nbqbits));
    int i = 0;
    int* temp = (int*)malloc(sizeof(int)*res.nbqbits);
    for (const auto& el: other.qbits){
        temp[i] = el;
        i++;
    }
    GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)res.qbits, temp, sizeof(int)*res.nbqbits, 0));
    i = 0;
    for (const auto& el: set<int>(other.qbits.begin(), other.qbits.end())){
        temp[i] = el;
        i++;
    }
    GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)res.ordered_qbits, temp, sizeof(int)*res.nbqbits, 0));
    GPU_CHECK(hipStreamAddCallback(0, freecallback, temp, 0));
    return res;
}

template<typename T>
GPUGate<T> createGPUGate(int n, vector<int> qbits){
    GPUGate<T> res;
    res.identifier = 0;
    res.optarg = 0;
    res.densecontent = createGPUMatrix<Complex<T>>((1llu << n));
    res.nbqbits = qbits.size();
    GPU_CHECK(hipMalloc(&res.qbits, sizeof(int)*res.nbqbits));
    GPU_CHECK(hipMalloc(&res.ordered_qbits, sizeof(int)*res.nbqbits));
    int i = 0;
    int* temp = (int*)malloc(sizeof(int)*res.nbqbits);
    for (const auto& el: qbits){
        temp[i] = el;
        i++;
    }
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)res.qbits, temp, sizeof(int)*res.nbqbits));
    i = 0;
    for (const auto& el: set<int>(qbits.begin(), qbits.end())){
        temp[i] = el;
        i++;
    }
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)res.ordered_qbits, temp, sizeof(int)*res.nbqbits));
    free(temp);
    return res;
}

template<typename T>
void destroyGPUGate(const GPUGate<T>& el){
    GPU_CHECK(hipFree(el.qbits));
    GPU_CHECK(hipFree(el.ordered_qbits));
    destroyGPUMatrix(el.densecontent);
}

template<typename T>
class QuantumCircuit;

template<typename T>
class GPUQuantumCircuit{
public:
    int gate_number;
    GPUGate<T>* gates;
    int nqbits;
};

template<typename T>
GPUQuantumCircuit<T> createGPUQuantumCircuit(const QuantumCircuit<T>& el){
    GPUQuantumCircuit<T> res;
    res.gate_number = el.gate_set_ordered.size();
    res.nqbits = el.nqbits;
    GPU_CHECK(hipMalloc(&res.gates, sizeof(GPUGate<T>)*res.gate_number));
    GPUGate<T>* temp = (GPUGate<T>*)malloc(sizeof(GPUGate<T>)*res.gate_number);
    for (int i = 0; i < res.gate_number; i++){
        temp[i] = createGPUGate<T>(el.gate_set_ordered[i]);
    }
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)res.gates, temp, sizeof(GPUGate<T>)*res.gate_number));
    free(temp);
    return res;
}

template<typename T>
GPUQuantumCircuit<T> createGPUQuantumCircuitAsync(const QuantumCircuit<T>& el){
    GPUQuantumCircuit<T> res;
    res.gate_number = el.gate_set_ordered.size();
    res.nqbits = el.nqbits;
    GPU_CHECK(hipMalloc(&res.gates, sizeof(GPUGate<T>)*res.gate_number));
    GPUGate<T>* temp = (GPUGate<T>*)malloc(sizeof(GPUGate<T>)*res.gate_number);
    for (int i = 0; i < res.gate_number; i++){
        temp[i] = createGPUGateAsync<T>(el.gate_set_ordered[i]);

    }
    GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)res.gates, temp, sizeof(GPUGate<T>)*res.gate_number, 0));
    GPU_CHECK(hipStreamAddCallback(0, freecallback, temp, 0));
    return res;
}

template<typename T>
void destroyGPUQuantumCircuit(const GPUQuantumCircuit<T>& el){
    GPUGate<T>* temp = (GPUGate<T>*)malloc(sizeof(GPUGate<T>)*el.gate_number);
    GPU_CHECK(hipMemcpyDtoH(temp, (hipDeviceptr_t)el.gates, sizeof(GPUGate<T>)*el.gate_number));
    GPU_CHECK(hipFree(el.gates));
    for (int i = 0; i < el.gate_number; i++){
        destroyGPUGate<T>(temp[i]);
    }
    free(temp);
}

//always use power of 2 for number of threads! powers of 2 are the best ;)
//yes, this kernel is pure madness and debugging will be worse than going to hell at least until you realize that you can walk out now if you d like.
//turns out it works and hell was elsewhere...
template<typename T>
__global__
void mergeGates(GPUGate<T> returnvalue, GPUQuantumCircuit<T> qc, int* coveredqbits_ordered, int gpunumberlog2, int gpuid, int sharedMemMatrixSize){ //the whole circuit is going to merge into a single matrix (that is the plan)
    int bit_to_groupbitnumber[64];
    for (int i = 0; i < qc.nqbits; i++){
        bit_to_groupbitnumber[coveredqbits_ordered[i]] = i;
    }
    
    size_t input = blockIdx.x + (gpuid << (qc.nqbits - gpunumberlog2));
    size_t initline = threadIdx.x;

    extern __shared__ Complex<T> qbitsstateANDmatrixstate[]; //must be of size sizeof(T)*2**nbqbits + sharedMemMatrixSize*sizeof(T)
    Complex<T>* qbitsstate = qbitsstateANDmatrixstate; //size sizeof(T)*2**nbqbits
    Complex<T>* matrixsharedstorage = qbitsstateANDmatrixstate + (1llu << qc.nqbits); //size sharedMemMatrixSize*sizeof(T)

    //initialization
    int work_per_thread0 = (1llu << qc.nqbits)/blockDim.x;
    for (int line = initline*work_per_thread0; line < (initline+1)*work_per_thread0; line++){
        qbitsstate[line] = Complex<T>(((line == input)? 1. : 0.), 0.);
    }
    __syncthreads(); //everyone in the block has fast acces to the whole state, now let s explore the circuit!

    for (int gateid = 0; gateid < qc.gate_number; gateid++){
        qc.gates[gateid].compute(qc.nqbits, qbitsstate, bit_to_groupbitnumber, matrixsharedstorage, sharedMemMatrixSize);
        __syncthreads();
    }
    //in theory we got our input colomn in qbitsstate and we just need to put it in vram at the right place
    for (int line = initline; line < initline+((1llu << qc.nqbits)/blockDim.x); line++){
        returnvalue.densecontent(line, input, qbitsstate[line]);
    }
}

template<typename T>
__host__ Gate<T> mergedGateMadness(vector<Gate<T>> to_merge, hipStream_t stream = 0){
    //first we need to not forget about reindexing
    int maxseen = 0;
    set<int> total_covered;
    for (const auto& gate: to_merge){
        for (const auto& qbit: gate.qbits){
            total_covered.insert(qbit);
            if (qbit > maxseen) maxseen = qbit;
        }
    }

    int* c_coveredqbits_ordered = (int*)malloc(sizeof(int)*total_covered.size());
    int* coveredqbits_ordered;
    int i = 0;
    for (const auto& el: total_covered){
        c_coveredqbits_ordered[i] = el;
        i++;
    }
    GPU_CHECK(hipMalloc(&coveredqbits_ordered, sizeof(int)*total_covered.size()));
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)coveredqbits_ordered, c_coveredqbits_ordered, sizeof(int)*total_covered.size()));
    /*
    //let's build permutation table
    vector<int> permuttable(maxseen+1);
    int i = 0;
    for (const auto& el: total_covered){
        permuttable[el] = i;
        i++;
    }
    vector<int> temp;
    for (auto& gate: to_merge){
        temp.clear();
        for (const auto& qbit: gate.qbits){
            temp.push_back(permuttable[qbit]);
        }
        gate.qbits = temp;
    }*/
    QuantumCircuit<T> to_mergecirc(to_merge, total_covered.size()); //temporary encapsulation to call the GPU one
    GPUQuantumCircuit<T> gpucircuit = createGPUQuantumCircuit<T>(to_mergecirc);
    //now the circuit is ready for inputing into the kernel
    //let's generate the returned GPUGate
    GPUGate<T> resGPU = createGPUGate<T>(total_covered.size(), vector<int>(total_covered.begin(), total_covered.end())); //the kernel will fill the matrix and these informations will be correct
    hipDeviceProp_t devattr;
    int device;
    GPU_CHECK(hipGetDevice(&device));
	GPU_CHECK(hipGetDeviceProperties(&devattr, device));
    size_t totalshared_block = devattr.sharedMemPerBlock;
    //only 1 gpu for now. If gpucircuit has less than 5 qbits, we are sad but it should work?
    //ideally, we should do it on CPU when nqbits < 8
    mergeGates<<<dim3((1llu << gpucircuit.nqbits)), dim3(min((int)1024, (int)(1llu << gpucircuit.nqbits))), totalshared_block, stream>>>(resGPU, gpucircuit, coveredqbits_ordered, 0, 0, (totalshared_block - (1llu << gpucircuit.nqbits))/sizeof(Complex<T>));
    Gate<T> res(resGPU);
    destroyGPUGate(resGPU);
    destroyGPUQuantumCircuit(gpucircuit);
    GPU_CHECK(hipFree(coveredqbits_ordered));
    free(c_coveredqbits_ordered);
    return res;
}

template<typename T>
class QuantumCircuit{
public:
    vector<Gate<T>> gate_set_ordered;
    int nqbits = 0;
    vector<pair<int, set<int>>> groups; //second is current qbit set, first is when to go to next group
    vector<int> initial_permutation;
    vector<int> final_inverse_permutation;
    vector<pair<int, vector<int>>> instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
    QuantumCircuit(int nqbits){
        this->nqbits = nqbits;
    }
    QuantumCircuit(const vector<Gate<T>>& gate_set, int nqbits){
        this->nqbits = nqbits;
        gate_set_ordered = gate_set;
    }
    void appendGate(const Gate<T>& gate){
        //checking qbits
        for (const auto& el: gate.qbits){
            if (el >= nqbits){
                cout << "the gate you are trying to add contains a qbit not in the circuit: " << el << "/" << nqbits << "!" << endl;
                return;
            }
        }
        if (groups.size() != 0) {
            cout << "you should not be adding gate after gateGrouping or allocate, this will not work" << endl;
        }
        gate_set_ordered.push_back(gate);
    }
    void print(bool permutationtable = 0){
        int i = 0;
        int group = 0;
        if (instructions.size() == 0){
            if (groups.size() != 0){
                cout << "Group 0 with qbits ";
                for (const auto& el2: groups[group].second){
                    cout << el2 << " ";
                }
                cout << " : " << endl;
            }
            for (auto& el: gate_set_ordered){
                if (groups.size() != 0){
                    if (group >= groups.size()){
                        cout << "ERROR while printing the circuit, group vector is bad" << endl;
                        return;
                    }
                    if (groups[group].first == i) {
                        group++;
                        cout << "Group " << group << " with qbits ";
                        for (const auto& el2: groups[group].second){
                            cout << el2 << " ";
                        }
                        cout << " : " << endl;
                    }
                    cout << "   ";
                    el.print();
                } else {
                    cout << "gate " << i << " is ";
                    el.print();
                }
                i++;
            }
        } else {
            vector<int> permutation = initial_permutation;
            vector<int> inversepermutation(nqbits);
            for (int m = 0; m < nqbits; m++){
                if (permutation[m] >= nqbits){
                    cout << "Error while printing: initial permutation work on more qbits than the number in the circuit!" << endl;
                    return;
                }
                inversepermutation[permutation[m]] = m;
            }

            //print when we did allocate
            cout << "-------Quantum-Program-------" << endl;
            for (const auto& instruction: instructions){
                if (instruction.first == 0){
                    cout << "SWAP ";
                    for (int m = 0; m < instruction.second.size()/2; m++){
                        cout << instruction.second[2*m] << " and " << instruction.second[2*m+1] << ", ";

                        swap(inversepermutation[instruction.second[2*m]], inversepermutation[instruction.second[2*m+1]]);
                        swap(permutation[inversepermutation[instruction.second[2*m]]], permutation[inversepermutation[instruction.second[2*m+1]]]);
                    }
                } else{
                    if (permutationtable){
                        cout << "Permutation table (subjective to real) : " << endl;
                        for (int m = 0; m < nqbits; m++){
                            cout << m << " ";
                            if (m < 10) cout << " ";
                        }
                        cout << endl;
                        for (int m = 0; m < nqbits; m++){
                            cout << inversepermutation[m] << " ";
                            if (inversepermutation[m] < 10) cout << " ";
                        }
                        cout << endl;
                        cout << endl;
                    }

                    cout << "EXEC Group " << group << " with qbits ";
                    for (const auto& el2: groups[group].second){
                        cout << el2 << " ";
                    }
                    cout << " : " << endl;
                    for (int j = i; j < groups[group].first; j++){
                        cout << "   ";
                        gate_set_ordered[j].print();
                    }
                    i = groups[group].first;
                    group++;
                }
                cout << endl << endl;
            }
            cout << "-----------------------------" << endl;
        }
    }
    void gateScheduling(){ //OPTIMISATION STEP 1 (non optimal)
        //this step is made to help the other optimisations do their jobs better by regrouping matrices on similar qbits
        if (gate_set_ordered.size() == 0) return;
        vector<vector<int>> dependencyGraph(gate_set_ordered.size());
        vector<vector<int>> dependencyGraphReversed(gate_set_ordered.size());
        Graph dependencyGraphMat(gate_set_ordered.size());

        int* lastused = (int*)malloc(sizeof(int)*nqbits);
        for (int i = 0; i < nqbits; i++){
            lastused[i] = -1;
        }
        for (int i = 0; i < gate_set_ordered.size(); i++){
            for (const auto& qbit: gate_set_ordered[i].qbits){
                if (lastused[qbit] == -1){
                    lastused[qbit] = i;
                } else {
                    if (dependencyGraphMat(lastused[qbit], i) == -1){
                        dependencyGraph[lastused[qbit]].push_back(i);
                        dependencyGraphReversed[i].push_back(lastused[qbit]);
                        dependencyGraphMat(lastused[qbit], i, 1);
                    } else {
                        dependencyGraphMat(lastused[qbit], i, 1+dependencyGraphMat(lastused[qbit], i));
                    }
                    lastused[qbit] = i;
                }
            }
        }
        //dependencyGraphMat.print();
        //check for each gate the depency constraints
        free(lastused);

        //now we need to use an euristic to explore the constructed DLG hence constructing the scheduling
        //the objective here is to regroup gates working on similar qbits as most as possible
        //an optimal way of doing this is to explore recursively from the end and .. in a way to test all combinations. However, this is not possible here
        //As such, we will execute a depth first search which favorise big connections locally (depth 1 search)
        vector<int> neworder;
        vector<bool> f(gate_set_ordered.size(), true);

        //get gates with on which no one relies on except us
        list<int> possible_nodes;
        for (int i = 0; i < gate_set_ordered.size(); i++){
            if (dependencyGraph[i].size() == 0){
                possible_nodes.push_back(i);
            }
        }

        vector<int> remaining_dependencies(gate_set_ordered.size());
        for (int i = 0; i < gate_set_ordered.size(); i++){
            remaining_dependencies[i] = dependencyGraph[i].size();
        }

        set<int> last_qbits = {};
        int current_weight;
        double temp;
        while (!possible_nodes.empty()){
            //we need to choose one of the possible nodes depending on last qbits
            auto itsaved = possible_nodes.begin();
            current_weight = -INT32_MAX;
            for (auto it = possible_nodes.begin(); it != possible_nodes.end(); it++){
                temp = /*last_qbits.size()*/ + gate_set_ordered[*it].qbits.size() - union_elements(last_qbits, gate_set_ordered[*it].qbits).size();
                if (temp > current_weight){
                    current_weight = temp;
                    itsaved = it;
                }
            }
            auto itsaved2 = itsaved;
            int u = *itsaved;
            itsaved2++;
            possible_nodes.erase(itsaved, itsaved2);

            neworder.push_back(u);
            last_qbits = set<int>(gate_set_ordered[u].qbits.begin(), gate_set_ordered[u].qbits.end());
            for (const auto& el: dependencyGraphReversed[u]){
                remaining_dependencies[el]--;
                if (remaining_dependencies[el] == 0){
                    possible_nodes.push_back(el);
                }
            }
        }

        //now we just need to reverse neworder and put the gates in the right order!
        vector<Gate<T>> newvect;
        for (int i = 0; i < neworder.size(); i++){
            newvect.push_back(gate_set_ordered[neworder[neworder.size()-1-i]]);
        }
        gate_set_ordered = newvect;
    }
    void gateFusion(int qbitsizelimit = 5, double merge_time_matters = 0.000001){ //OPTIMISATION STEP 2 (optimal given a schedule)
        //merge_time_matters takes into account the compilation time as a factor. ideal value is 1/(expected execution per compilation). 0 for best possible compilation
        //qbitsizelimit should ideally stay below shared_memory capacity as stated in printGPUInfo
        //this step is to reduce both memory bandwidth pressure and computation time pressure
        //memory bandwidth pressure should not be taken into account in defining merge_time_matters since the problem will be dealt with at opt step 3 anyway
        set<int> precompute;
        double temp;
        double temp2;
        Graph optim(gate_set_ordered.size()+1); //from 0 (no execution) to number_of_gate included for everything executed
        for (int i = 0; i < gate_set_ordered.size(); i++){
            precompute = set<int>(gate_set_ordered[i].qbits.begin(), gate_set_ordered[i].qbits.end());
            switch (gate_set_ordered[i].identifier){
                case 0:
                    temp = (1llu << precompute.size());
                    optim(i, i+1, temp);
                    break;
                case Hadamard:
                    temp = 2;
                    optim(i, i+1, temp);
                    break;
                case CNOT:
                    temp = 0.75;
                    optim(i, i+1, temp);
                    break;
                case CRk:
                    temp = 0.25;
                    optim(i, i+1, temp);
                    break;
                case TOFFOLI:
                    temp = (double)3/8;
                    optim(i, i+1, temp);
                    break;
            }
            temp2 = 0;
            for (int j = i+1; j < gate_set_ordered.size(); j++){
                precompute = union_elements(precompute, gate_set_ordered[j].qbits);
                if (precompute.size() > qbitsizelimit) break;
                temp2 += (1llu << gate_set_ordered[j].qbits.size()); //precomputed value for merge_time theoric estimation
                //no more switch case now it s always dense (but it can be improved with more intensive precomputing)
                temp = (1llu << precompute.size());
                temp += merge_time_matters*temp2*(1llu << (2*precompute.size()))/(1llu << nqbits); //cost of merging
                optim(i, j+1, temp);
            }
        }
        //now we just need to path find from 0 to nqbits to know where to merge!
        vector<int> path = optim.pathFinding(0, gate_set_ordered.size());
        //let's merge
        vector<Gate<T>> newcircuit;
        int f = -1;
        for (const auto& el: path){
            if (f == -1) {
                f = el;
                continue;
            }
            //cout << "merge " << f << " up to " << el << endl;
            if (el == f+1){
                newcircuit.push_back(gate_set_ordered[f]);
                f = el;
                continue;
            }
            //we need to merge gates from f (included) to el (excluded) which consist in the evaluation of the output for each input
            //cost of a merge of k gates is at worst k*(2**nqbits)**3 which can be worse than executing the circuit once when nqbits is too high. as such, it is recommended to limit the size of the merge to at most nqbits/3 except if the circuit is gonna be reused a lot
            //for that purpose, we could rerun the simulator on each possible input.
            vector<Gate<T>> to_merge;
            for (int i = f; i < el; i++){
                to_merge.push_back(gate_set_ordered[i]);
            }
            Gate<T> ngate = mergedGateMadness<T>(to_merge);

            newcircuit.push_back(ngate);

            f = el;
        }
        gate_set_ordered = newcircuit;
    }
    void gateGrouping(int qbitgroupsize = 10){ // OPTIMISATION STEP 3 (optimal given a schedule)
        //this step is exclusively to reduce memory bandwidth pressure
        //qbitgroupsize should be set to what your registers per thread can tolere (minus some margin for the overhead registers)
        set<int> precompute;
        Graph optim(gate_set_ordered.size()+1);
        for (int i = 0; i < gate_set_ordered.size(); i++){
            precompute.clear();
            for (int j = i; j < gate_set_ordered.size(); j++){
                precompute = union_elements(precompute, gate_set_ordered[j].qbits);
                if (precompute.size() > qbitgroupsize) break; //went off limit
                optim(i, j+1, 1);
            }
        }
        //now the groups will be given by path finding!
        vector<int> fgroups = optim.pathFinding(0, gate_set_ordered.size());
        groups = vector<pair<int, set<int>>>(fgroups.size()-1); // no 0
        //let's precompute qbits of each groups
        int group = 1;
        precompute.clear();
        for (int i = 0; i < gate_set_ordered.size(); i++){
            if (fgroups[group] == i){
                groups[group-1].first = i;
                groups[group-1].second = precompute;
                precompute.clear();
                group++;
            }
            precompute = union_elements(precompute, gate_set_ordered[i].qbits);
        }
        groups[groups.size()-1].first = gate_set_ordered.size();
        groups[groups.size()-1].second = precompute;
    }
    void allocate(int numberofgpulog2 = 0, int maxlocalqbitnumber= 300){ //OPTIMISATION STEP 4 (it can be further optimised taking into account multiple swaps at the same time)
        //only support homogeneous gpus or the slow one will slow the big one
        if (maxlocalqbitnumber + numberofgpulog2 < nqbits){
            cout << "Error: Can't allocate - Too much qbits in the circuit to handle with " << maxlocalqbitnumber << " localqbits and " << (1llu << numberofgpulog2) << " gpus" << endl;
            return;
        }
        
        maxlocalqbitnumber = nqbits - numberofgpulog2; //this line is to modify the behaviour from "use fewest amount of gpu to as much as permitted by options"
        //comment this line to come back to old behaviour

        instructions = {};
        //if no grouping optimisation is done, we will use naive grouping which is one group per gate because it is necessary for our later processing
        if (groups.size() == 0){
            for (int i = 0; i < gate_set_ordered.size(); i++){
                groups.push_back(make_pair(i+1, set<int>(gate_set_ordered[i].qbits.begin(), gate_set_ordered[i].qbits.end())));
            }
        }
        //if (nqbits <= maxlocalqbitnumber || numberofgpulog2 == 0){
        //    //just need to push compute1 number of group times
        //    for (int i = 0; i < groups.size(); i++){
        //        instructions.push_back(make_pair(1, vector<int>()));
        //    }
        //    return;
        //}
        //we need to know at each step when will a qbit be useful next. 
        //A way to do it in linear time is to precompute when it is used when it will be used next which can be done in linear time
        //there is complicated but doable way of doing it in nlogn total but here we will see a n**2 way with the precomputation
        vector<set<pair<int, int>>> precompute(groups.size()); //pair<int,int> is (qbit, time before reappearing)
        vector<int> last_seen(nqbits, INT32_MAX); //you wouldn't use anywhere close to 2**32 gates right?
        for (int i = groups.size()-1; i >= 0; i--){
            for (const auto& el: groups[i].second){ //all qbit of a group
                precompute[i].insert(make_pair(el, last_seen[el] - i));
                last_seen[el] = i;
            }
        }
        //now we can start allocating in the direct direction instead of the reverse one like the precomputation

        //first is the initialization using.. the remaining unused end state of last_seen!
        vector<int> last_seenid(last_seen.size());
        for (int i = 0; i < nqbits; i++){
            last_seenid[i] = i;
        } //we will sort the array so this is useful to remember indexes
        sort(last_seenid.begin(), last_seenid.end(), [&last_seen](int a, int b){return last_seen[a] < last_seen[b];});
        vector<int> locals, globals;
        locals = vector<int>(last_seenid.begin(),last_seenid.end()-numberofgpulog2);
        globals = vector<int>(last_seenid.end()-numberofgpulog2, last_seenid.end());
        //last part of initialization
        vector<int> nextsee(nqbits, 0);
        vector<int> permutation(nqbits, 0); //super important
        vector<int> inversepermutation(nqbits, 0);
        //qbit is local if permutation[qbit] < maxlocalqbitnumber
        for (int i = 0; i < nqbits; i++){
            nextsee[i] = last_seen[i];
        }

        int i = 0;
        for (const auto& el: locals){
            permutation[el] = i; //permutation is real to virtual
            inversepermutation[i] = el; //virtual to real
            i++;
        }
        for (const auto& el: globals){
            permutation[el] = i;
            inversepermutation[i] = el;
            i++;
        }

        initial_permutation = permutation;

        //i <-> j, permutation[i] <-> permutation[j]
        //now we can definitely generate instructions!
        vector<pair<int, int>> pairs;
        set<int> alreadytaken;
        int k = 0; //gate index
        for (int i = 0; i < groups.size(); i++){
            for (int l = 0; l < nqbits; l++){
                nextsee[l] -= 1;
            }
            pairs = {};
            alreadytaken = set<int>(groups[i].second.begin(), groups[i].second.end());
            for (const auto& el: groups[i].second){ //let's check who we need to swap!
                if (permutation[el] >= maxlocalqbitnumber){
                    //ho no you are in global!
                    int worstqbit = -1;
                    for (int j = 0; j < maxlocalqbitnumber; j++){
                        if (alreadytaken.find(inversepermutation[j]) != alreadytaken.end()){
                            continue;
                        }
                        if (worstqbit == -1){
                            worstqbit = inversepermutation[j];
                            continue;
                        }
                        if (nextsee[inversepermutation[j]] > nextsee[worstqbit]){
                            worstqbit = inversepermutation[j];
                        }
                    }

                    if (worstqbit == -1){
                        cout << "ALLOCATION FAILED: not enough localqbit available for a given group" << endl;
                        return;
                    }
                    //beware that pairs take into account permutations that have already happened!
                    
                    pairs.push_back(make_pair(permutation[el], permutation[worstqbit]));
                    //now let's refresh permutations
                    swap(inversepermutation[permutation[el]], inversepermutation[permutation[worstqbit]]);
                    swap(permutation[el], permutation[worstqbit]);
                }
                nextsee[el] = INT32_MAX; //temporary, we will update right outside the loop
            }
            for (const auto& refreshpair: precompute[i]){
                nextsee[refreshpair.first] = refreshpair.second;
            }
            if (pairs.size() != 0){
                //swap operation needed!
                vector<int> pairsset;
                for (const auto& pair: pairs){
                    pairsset.push_back(pair.first);
                    pairsset.push_back(pair.second);
                }
                instructions.push_back(make_pair(0, pairsset));
            }
            instructions.push_back(make_pair(1, vector<int>()));
            //we need to modify gates subjective qbits and of the group
            set<int> temp;
            for (const auto& el: groups[i].second){
                temp.insert(permutation[el]);
            }
            groups[i].second = temp;
            vector<int> temp2;
            for (int l = k; l < groups[i].first; l++){
                temp2.clear();
                for (const auto& qbit: gate_set_ordered[l].qbits){
                    temp2.push_back(permutation[qbit]);
                }
                gate_set_ordered[l].qbits = temp2;
            }
            k = groups[i].first;
        }

        final_inverse_permutation = inversepermutation;
    }
    void compileOPT(int qbit_matrix_merge_size_limit = 5, double merge_time_matters = 0.000001, int groupsize = 10, int numberofgpulog2 = 0, int maxlocalqbitnumber = 300){
        gateScheduling();
        gateFusion(qbit_matrix_merge_size_limit, merge_time_matters);
        gateGrouping(groupsize);
        allocate(numberofgpulog2, maxlocalqbitnumber); //will try to use every gpus you give it! sometimes it is not worth it
    }
    void compileDefault(int numberofgpulog2 = 0, int maxlocalqbitnumber = 300){ //for every optimization that hasnt been done but was necessary, it will use naive things to replace them
        //only support homogeneous gpus or the slow one will slow the big one
        if (instructions.size() != 0) return; // case where everything has already been done (possible that allocate was optimized but not grouping)
        
        if (maxlocalqbitnumber + numberofgpulog2 < nqbits){
            cout << "Error: Can't allocate - Too much qbits in the circuit to handle with " << maxlocalqbitnumber << " localqbits and " << (1llu << numberofgpulog2) << " gpus" << endl;
            return;
        }
        
        maxlocalqbitnumber = nqbits - numberofgpulog2; //this line is to modify the behaviour from "use fewest amount of gpu to as much as permitted by options"
        //comment this line to come back to old behaviour
        if (groups.size() == 0){
            for (int i = 0; i < gate_set_ordered.size(); i++){
                groups.push_back(make_pair(i+1, set<int>(gate_set_ordered[i].qbits.begin(), gate_set_ordered[i].qbits.end())));
            }
        }
        //now we need the naive allocation (allocate necessary qbit to the most left qbit)
        vector<int> permutation(nqbits, 0); //super important
        vector<int> inversepermutation(nqbits, 0);
        for (int i = 0; i < nqbits; i++){
            permutation[i] = i;
            inversepermutation[i] = i;
        }

        initial_permutation = permutation;

        int k = 0;
        set<int> alreadytaken;
        vector<pair<int, int>> pairs;
        for (int group = 0; group < groups.size(); group++){
            pairs = {};
            alreadytaken = groups[group].second;
            //let s find the leftmost free for each globalqbit that we need
            for (const auto& qbit: groups[group].second){
                if (permutation[qbit] >= maxlocalqbitnumber){
                    //ho no! our qbit is non local
                    int newlocal;
                    for (int j = 0; j < maxlocalqbitnumber; j++){
                        if (alreadytaken.find(inversepermutation[j]) != alreadytaken.end()){
                            continue;
                        }
                        newlocal = inversepermutation[j];
                        break;
                    }
                    //beware that pairs take into account permutations that have already happened!
                    pairs.push_back(make_pair(permutation[qbit], permutation[newlocal]));
                    //now let's refresh permutations
                    swap(inversepermutation[permutation[qbit]], inversepermutation[permutation[newlocal]]);
                    swap(permutation[qbit], permutation[newlocal]);
                }
            }
            if (pairs.size() != 0){
                //swap operation needed!
                vector<int> pairsset;
                for (const auto& pair: pairs){
                    pairsset.push_back(pair.first);
                    pairsset.push_back(pair.second);
                }
                instructions.push_back(make_pair(0, pairsset));
            }
            instructions.push_back(make_pair(1, vector<int>()));
            //we need to modify gates subjective qbits and of the group
            set<int> temp;
            for (const auto& el: groups[group].second){
                temp.insert(permutation[el]);
            }
            groups[group].second = temp;
            vector<int> temp2;
            for (int l = k; l < groups[group].first; l++){
                temp2.clear();
                for (const auto& qbit: gate_set_ordered[l].qbits){
                    temp2.push_back(permutation[qbit]);
                }
                gate_set_ordered[l].qbits = temp2;
            }
            k = groups[group].first;
        }
        final_inverse_permutation = inversepermutation;
    }
    void dual_phase_allocation(int gpu_per_node_log2, int nodelog2){ //will produce global global swaps (from slow and fast qbits)
        QuantumCircuit<T> we = *this;
        we.allocate(nodelog2); //first consider the allocation with global qbits nodelog2
        QuantumCircuit<T> res(we.gate_set_ordered, nqbits-nodelog2); //virtually, this new circuit works on only fastqbits
        res.groups = we.groups; //groups are the same
        //WE SHOULD NOT OPTIMIZE THIS CIRCUIT except allocation
        res.allocate(gpu_per_node_log2); //this time we allocate with the real local qbits

        //now we need to adapt these results to our own circuit: restore initial and final permutation, put swap commands, get back data
        groups = res.groups; //we take the last subjective version (without any non local qbits)
        gate_set_ordered = res.gate_set_ordered;
        //qbits_number is not touched
        initial_permutation = vector<int>(nqbits);
        final_inverse_permutation = vector<int>(nqbits);

        for (int i = nqbits-nodelog2; i < nqbits; i++){
            initial_permutation[i] = we.initial_permutation[i];
            final_inverse_permutation[i] = we.final_inverse_permutation[i];
        }
        for (int i = 0; i < nqbits; i++){
            if (we.initial_permutation[i] < nqbits-nodelog2){
                initial_permutation[i] = res.initial_permutation[we.initial_permutation[i]];
            } else { //we start in global so we are not modified by res
                initial_permutation[i] = we.initial_permutation[i];
            }
            if (i < nqbits-nodelog2){
                final_inverse_permutation[i] = we.final_inverse_permutation[res.final_inverse_permutation[i]];
            } else {//i is not in res managment
                final_inverse_permutation[i] = we.final_inverse_permutation[i];
            }
        }
        //we need to change the swap order from we to make them subjective with respect to res this requires going through all the instructions and keeping the permutation table of res.
        instructions = {};
        vector<int> pairsset;
        vector<int> we_to_res = res.initial_permutation;
        vector<int> res_to_we(nqbits-nodelog2);
        for (int i = 0; i < nqbits-nodelog2; i++){
            res_to_we[we_to_res[i]] = i;
        }
        int instridres = 0;
        int instridwe = 0;
        for (int groupid = 0; groupid < groups.size(); groupid++){
            while (we.instructions[instridwe].first != 1){
                //we got a swap command to transform
                pairsset = {};
                for (const auto& el: we.instructions[instridwe].second){
                    if (el >= nqbits-nodelog2) {pairsset.push_back(el); continue;}
                    pairsset.push_back(we_to_res[el]);
                }
                instructions.push_back(make_pair(0, pairsset));
                instridwe++;
            }
            instridwe++;
            while (res.instructions[instridres].first != 1){
                //we got a res swap command to refresh we_to_res
                for (int i = 0; i+1 < res.instructions[instridres].second.size(); i += 2){
                    int first = res.instructions[instridres].second[i];
                    int second = res.instructions[instridres].second[i+1];
                    //swap first and second (they are subjective to res)
                    swap(res_to_we[first], res_to_we[second]);
                    swap(we_to_res[res_to_we[first]], we_to_res[res_to_we[second]]);
                }
                instructions.push_back(res.instructions[instridres]);
                instridres++;
            }
            instridres++;
            //finally we can add the command execution
            instructions.push_back(make_pair(1, vector<int>()));
            
        }
    }
    void slow_fast_allocation(int fastqbits, int slowqbits, double fasttime, double slowtime){
        instructions = {};
        //if no grouping optimisation is done, we will use naive grouping which is one group per gate because it is necessary for our later processing
        if (groups.size() == 0){
            for (int i = 0; i < gate_set_ordered.size(); i++){
                groups.push_back(make_pair(i+1, set<int>(gate_set_ordered[i].qbits.begin(), gate_set_ordered[i].qbits.end())));
            }
        }
        //if (nqbits <= maxlocalqbitnumber || numberofgpulog2 == 0){
        //    //just need to push compute1 number of group times
        //    for (int i = 0; i < groups.size(); i++){
        //        instructions.push_back(make_pair(1, vector<int>()));
        //    }
        //    return;
        //}
        //we need to know at each step when will a qbit be useful next. 
        //A way to do it in linear time is to precompute when it is used when it will be used next which can be done in linear time
        //there is complicated but doable way of doing it in nlogn total but here we will see a n**2 way with the precomputation
        vector<set<pair<int, int>>> precompute(groups.size()); //pair<int,int> is (qbit, time before reappearing)
        vector<int> last_seen(nqbits, INT32_MAX); //you wouldn't use anywhere close to 2**32 gates right?
        for (int i = groups.size()-1; i >= 0; i--){
            for (const auto& el: groups[i].second){ //all qbit of a group
                precompute[i].insert(make_pair(el, last_seen[el] - i));
                last_seen[el] = i;
            }
        }
        //now we can start allocating in the direct direction instead of the reverse one like the precomputation

        //first is the initialization using.. the remaining unused end state of last_seen!
        vector<int> last_seenid(last_seen.size());
        for (int i = 0; i < nqbits; i++){
            last_seenid[i] = i;
        } //we will sort the array so this is useful to remember indexes
        sort(last_seenid.begin(), last_seenid.end(), [&last_seen](int a, int b){return last_seen[a] < last_seen[b];});
        vector<int> locals, fasts, slows;
        locals = vector<int>(last_seenid.begin(),last_seenid.end()-slowqbits-fastqbits);
        fasts = vector<int>(last_seenid.end()-slowqbits-fastqbits, last_seenid.end()-slowqbits);
        slows = vector<int>(last_seenid.end()-slowqbits, last_seenid.end());
        //last part of initialization
        vector<int> nextsee(nqbits, 0);
        vector<int> permutation(nqbits, 0); //super important
        vector<int> inversepermutation(nqbits, 0);
        //qbit is local if permutation[qbit] < maxlocalqbitnumber
        for (int i = 0; i < nqbits; i++){
            nextsee[i] = last_seen[i]+1;
        }

        int i = 0;
        for (const auto& el: locals){
            permutation[el] = i; //permutation is real to virtual
            inversepermutation[i] = el; //virtual to real
            i++;
        }
        for (const auto& el: fasts){
            permutation[el] = i;
            inversepermutation[i] = el;
            i++;
        }
        for (const auto& el: slows){
            permutation[el] = i;
            inversepermutation[i] = el;
            i++;
        }

        initial_permutation = permutation;

        //i <-> j, permutation[i] <-> permutation[j]
        //now we can definitely generate instructions!
        vector<pair<int, int>> pairs;
        set<int> alreadytaken;
        int k = 0; //gate index
        for (int i = 0; i < groups.size(); i++){
            for (int l = 0; l < nqbits; l++){
                nextsee[l] -= 1;
            }
            pairs = {};
            alreadytaken = set<int>(groups[i].second.begin(), groups[i].second.end());
            for (const auto& el: groups[i].second){ //let's check who we need to swap!
                if (permutation[el] >= nqbits-slowqbits-fastqbits && permutation[el] < nqbits-slowqbits){
                    //you are in the fast qbit cache! we need to swap, finding the best local qbit for that but eventually, the local one might have its place in slow so we will need to tackle this
                    int worstqbit = -1;
                    for (int j = 0; j < nqbits-slowqbits-fastqbits; j++){
                        if (alreadytaken.find(inversepermutation[j]) != alreadytaken.end()){
                            continue;
                        }
                        if (worstqbit == -1){
                            worstqbit = inversepermutation[j];
                            continue;
                        }
                        if (nextsee[inversepermutation[j]] > nextsee[worstqbit]){
                            worstqbit = inversepermutation[j];
                        }
                    }

                    if (worstqbit == -1){
                        cout << "ALLOCATION FAILED: not enough localqbit available for a given group" << endl;
                        return;
                    }

                    //beware that pairs take into account permutations that have already happened!
                    
                    pairs.push_back(make_pair(permutation[el], permutation[worstqbit]));
                    //now let's refresh permutations
                    swap(inversepermutation[permutation[el]], inversepermutation[permutation[worstqbit]]);
                    swap(permutation[el], permutation[worstqbit]);
                } else if (permutation[el] >= nqbits-slowqbits){
                    //ho no, you are a slow qbits! there are only 2 options: going to the fast cache first then local, or local directly
                    int worstqbit = -1;
                    double weight;
                    for (int j = 0; j < nqbits-slowqbits-fastqbits; j++){ //let's investigate the direct swap weight
                        if (alreadytaken.find(inversepermutation[j]) != alreadytaken.end()){
                            continue;
                        }
                        if (worstqbit == -1){
                            worstqbit = inversepermutation[j];
                            weight = nextsee[worstqbit];
                            continue;
                        }
                        if (nextsee[inversepermutation[j]] > weight){
                            worstqbit = inversepermutation[j];
                            weight = nextsee[worstqbit];
                        }
                    }

                    int bestlocal = worstqbit;
                    weight = slowtime/weight; //we need to minimize this
                    //cout << "starting phase" << endl;
                    //cout << nextsee[worstqbit] << endl;

                    for (int j = nqbits-slowqbits-fastqbits; j < nqbits-slowqbits; j++){ //let's investigate the cache
                        if (alreadytaken.find(inversepermutation[j]) != alreadytaken.end()){
                            continue;
                        }
                        if (worstqbit == -1){
                            worstqbit = inversepermutation[j];
                            weight = slowtime/(double)nextsee[worstqbit] + fasttime;
                            continue;
                        }
                        //cout << nextsee[inversepermutation[j]] << endl;
                        if ((slowtime/(double)nextsee[inversepermutation[j]]) + fasttime < weight){
                            worstqbit = inversepermutation[j];
                            weight = slowtime/(double)nextsee[worstqbit] + fasttime;
                        }
                    }
                    //cout << "end phase" << endl;

                    if (worstqbit == -1){
                        cout << "ALLOCATION FAILED: not enough localqbit available for a given group" << endl;
                        return;
                    }
                    //beware that pairs take into account permutations that have already happened!
                    
                    //let's swap the found qbit
                    pairs.push_back(make_pair(permutation[el], permutation[worstqbit]));
                    //now let's refresh permutations
                    swap(inversepermutation[permutation[el]], inversepermutation[permutation[worstqbit]]);
                    swap(permutation[el], permutation[worstqbit]);

                    //if the found qbit was in cache, we also need to swap this new fast qbit with the best local candidate that we saved
                    if (permutation[el] >= nqbits-fastqbits-slowqbits){
                        pairs.push_back(make_pair(permutation[bestlocal], permutation[el]));
                        //now let's refresh permutations
                        swap(inversepermutation[permutation[bestlocal]], inversepermutation[permutation[el]]);
                        swap(permutation[bestlocal], permutation[el]);
                    }
                }
                nextsee[el] = INT32_MAX; //temporary, we will update right outside the loop
            }
            for (const auto& refreshpair: precompute[i]){
                nextsee[refreshpair.first] = refreshpair.second;
            }
            if (pairs.size() != 0){
                //swap operation needed!
                vector<int> pairsset;
                for (const auto& pair: pairs){
                    pairsset.push_back(pair.first);
                    pairsset.push_back(pair.second);
                }
                instructions.push_back(make_pair(0, pairsset));
            }
            instructions.push_back(make_pair(1, vector<int>()));
            //we need to modify gates subjective qbits and of the group
            set<int> temp;
            for (const auto& el: groups[i].second){
                temp.insert(permutation[el]);
            }
            groups[i].second = temp;
            vector<int> temp2;
            for (int l = k; l < groups[i].first; l++){
                temp2.clear();
                for (const auto& qbit: gate_set_ordered[l].qbits){
                    temp2.push_back(permutation[qbit]);
                }
                gate_set_ordered[l].qbits = temp2;
            }
            k = groups[i].first;
        }

        final_inverse_permutation = inversepermutation;
    }
};