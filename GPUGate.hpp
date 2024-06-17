#ifndef GPUGATEDONE
#define GPUGATEDONE

#include "preprocessor.hpp"
#include "basic_host_types.hpp"

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

#endif