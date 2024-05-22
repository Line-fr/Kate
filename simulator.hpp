#include "gate.hpp"
#include<chrono>

#define USE_PEER_NON_UNIFIED 1
#include<hip/hip_runtime.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

template<typename T>
__global__ void initialize_state(int nqbits, Complex<T>* memory, int indexfor1){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    int work_per_thread = (1llu << nqbits)/blockDim.x/gridDim.x;

    for (size_t i = tid*work_per_thread; i < (tid+1)*work_per_thread; i++){
        memory[i] = (i == indexfor1)? 1 : 0;
    }
}

template<typename T>
__global__ void initialize_state0(int nqbits, Complex<T>* memory){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    int work_per_thread = (1llu << nqbits)/blockDim.x/gridDim.x;

    for (size_t i = tid*work_per_thread; i < (tid+1)*work_per_thread; i++){
        memory[i] = 0;
    }
}

template<typename T> 
__global__ void swapqbitKernelDirectAcess(int nqbits, int localq, Complex<T>* memory0, Complex<T>* memory1, int baseindex, int values_per_thread){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x + baseindex;

    size_t mask = (1llu << localq) - 1;
    size_t mask2 = (1llu << (nqbits - 1)) - 1 - mask;
    for (int i = tid*values_per_thread; i < (tid+1)*(values_per_thread); i++){
        size_t baseIndex = (i&mask) + ((i&mask2) << 1);
        Complex<T> temp = memory1[baseIndex]; //we want the 0 of device which has the global 1;
        memory1[baseIndex] = memory0[baseIndex + (1llu << localq)]; //then we paste the 1 of the device which has the global 0
        memory0[baseIndex + (1llu << localq)] = temp;
    }
}

template<typename T>
__global__ void swapqbitKernelIndirectAccessEXTRACT(int nqbits, int localq, size_t qbitvalue, Complex<T>* mymemory, Complex<T>* buffer, int baseindex, int values_per_thread){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x + baseindex;

    size_t mask = (1llu << localq) - 1;
    size_t mask2 = (1llu << (nqbits - 1)) - 1 - mask;

    size_t bufferbeg = values_per_thread*(threadIdx.x + blockIdx.x*blockDim.x);
    int p = 0;

    for (int i = tid*values_per_thread; i < (tid+1)*(values_per_thread); i++){
        size_t baseIndex = (i&mask) + ((i&mask2) << 1);
        
        buffer[bufferbeg+p] = mymemory[baseIndex + ((qbitvalue) << localq)]; 
        p++;
    }
}

template<typename T>
__global__ void swapqbitKernelIndirectAccessIMPORT(int nqbits, int localq, size_t qbitvalue, Complex<T>* mymemory, Complex<T>* buffer, int baseindex, int values_per_thread){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x + baseindex;

    size_t mask = (1llu << localq) - 1;
    size_t mask2 = (1llu << (nqbits - 1)) - 1 - mask;

    size_t bufferbeg = values_per_thread*(threadIdx.x + blockIdx.x*blockDim.x);
    int p = 0;

    for (int i = tid*values_per_thread; i < (tid+1)*(values_per_thread); i++){
        size_t baseIndex = (i&mask) + ((i&mask2) << 1);
        
        mymemory[baseIndex + ((qbitvalue) << localq)] = buffer[bufferbeg+p]; 
        p++;
    }
}

template<typename T>
__global__ void executeGroupKernelSharedState(int nqbits, Complex<T>* qbitsstate, int groupnqbits, int* groupqbits, GPUGate<T>* gates, int gatenumber, int sharedMemMatrixSize){
    int bit_to_groupbitnumber[64];
    for (int i = 0; i < groupnqbits; i++){
        bit_to_groupbitnumber[groupqbits[i]] = i;
    }

    size_t groupel = blockIdx.x;
    size_t initline = threadIdx.x;

    extern __shared__ Complex<T> qbitsstateANDmatrixstate[]; //must be of size sizeof(T)*2**nbqbits + sharedMemMatrixSize
    Complex<T>* qbitsstateshared = qbitsstateANDmatrixstate; //size 2**(groupnqbits)
    Complex<T>* matrixsharedstorage = qbitsstateANDmatrixstate + (1llu << groupnqbits); //size sharedMemMatrixSize

    size_t mask_group[64];
    size_t cumulative = 0;
    for (int i = 0; i < groupnqbits; i++){
        mask_group[i] = (1llu << (groupqbits[i] - i)) - 1 - cumulative;
        cumulative += mask_group[i];
    }
    mask_group[groupnqbits] = (1llu << (nqbits - groupnqbits)) - 1 - cumulative;


    size_t groupbaseind = 0;
    for (int i = 0; i <= groupnqbits; i++){
        groupbaseind += ((groupel&mask_group[i]) << i);
    } // XXXX-0-XXXXX-0-XX... 0 for all qbit group of the group
    //initialization
    int work_per_thread0 = (1llu << groupnqbits)/blockDim.x;
    if (work_per_thread0 == 0 && threadIdx.x < (1llu << groupnqbits)){
        size_t finalbaseind = groupbaseind;
        for (int i = 0; i < groupnqbits; i++){
            finalbaseind += ((initline >> i)%2) << groupqbits[i];
        }

        qbitsstateshared[initline] = qbitsstate[finalbaseind];
    }
    
    for (int line = initline*work_per_thread0; line < (initline+1)*work_per_thread0; line++){
        size_t finalbaseind = groupbaseind;
        for (int i = 0; i < groupnqbits; i++){
            finalbaseind += ((line >> i)%2) << groupqbits[i];
        }
        
        qbitsstateshared[line] = qbitsstate[finalbaseind];
        //printf("value at line: %i is %f with finalbaseind : %i\n", line, qbitsstateshared[line].a, (int)finalbaseind);
    }

    __syncthreads(); //everyone in the block has fast access to the whole group state, now let s explore the circuit!

    for (int gateid = 0; gateid < gatenumber; gateid++){
        gates[gateid].compute(groupnqbits, qbitsstateshared, bit_to_groupbitnumber, matrixsharedstorage, sharedMemMatrixSize);
        __syncthreads();
    }

    if (work_per_thread0 == 0 && threadIdx.x < (1llu << groupnqbits)){
        size_t finalbaseind = groupbaseind;
        for (int i = 0; i < groupnqbits; i++){
            finalbaseind += ((initline >> i)%2) << groupqbits[i];
        }

        qbitsstate[finalbaseind] = qbitsstateshared[initline];
    }
    for (int line = initline*work_per_thread0; line < (initline+1)*work_per_thread0; line++){
        size_t finalbaseind = groupbaseind;
        for (int i = 0; i < groupnqbits; i++){
            finalbaseind += ((line >> i)%2) << groupqbits[i];
        }

        qbitsstate[finalbaseind] = qbitsstateshared[line];
    }

    //__syncthreads();
}

template<typename T>
class Simulator{
public:
    //copy from quantum circuit but with the gpu version
    vector<pair<int, set<int>>> groups; //second is current qbit set, first is when to go to next group
    vector<int> initial_permutation;
    vector<int> final_inverse_permutation;
    vector<pair<int, vector<int>>> instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
    vector<Gate<T>> gate_set_ordered;
    int nqbits = 0;

    int number_of_gpu;
    int number_of_gpu_log2;
    Complex<T>** gpu_qbits_states;
    GPUQuantumCircuit<T>* gpuc; //one for each device

    Complex<T>** swapBuffer1 = NULL;
    Complex<T>** swapBuffer2 = NULL;
    Complex<T>** cswapBuffer = NULL;
    int swapBufferSizeLog2;

    Simulator(QuantumCircuit<T> mycircuit, int number_of_gpu, int swapBufferSizeLog2 = 24){
        if (mycircuit.instructions.size() == 0){
            cout << "warning: the simulator has been input a circuit that is not compiled. I will compile it naively now" << endl;
            mycircuit.compileDefault((int)log2(number_of_gpu), mycircuit.nqbits - (int)log2(number_of_gpu));
        }
        groups = mycircuit.groups;
        initial_permutation = mycircuit.initial_permutation;
        final_inverse_permutation = mycircuit.final_inverse_permutation;
        instructions = mycircuit.instructions;
        gate_set_ordered = mycircuit.gate_set_ordered;
        nqbits = mycircuit.nqbits;

        number_of_gpu_log2 = (int)log2(number_of_gpu);
        this->number_of_gpu = (1llu << number_of_gpu_log2);
        gpu_qbits_states = (Complex<T>**)malloc(sizeof(Complex<T>*)*number_of_gpu);
        if (number_of_gpu > 1) swapBuffer1 = (Complex<T>**)malloc(sizeof(Complex<T>*)*number_of_gpu);
        if (number_of_gpu > 1) swapBuffer2 = (Complex<T>**)malloc(sizeof(Complex<T>*)*number_of_gpu);
        if (number_of_gpu > 1) cswapBuffer = (Complex<T>**)malloc(sizeof(Complex<T>*)*number_of_gpu);
        
        this->swapBufferSizeLog2 = swapBufferSizeLog2;
        gpuc = (GPUQuantumCircuit<T>*)malloc(sizeof(GPUQuantumCircuit<T>)*number_of_gpu);
        for (int i = 0; i < number_of_gpu; i++){
            if (number_of_gpu > 1) cswapBuffer[i] = (Complex<T>*)malloc(sizeof(Complex<T>)*(1llu << swapBufferSizeLog2));
            hipSetDevice(i);
            hipMalloc(gpu_qbits_states+i, sizeof(Complex<T>)*(1llu << (nqbits - number_of_gpu_log2)));
            if (number_of_gpu > 1) hipMalloc(swapBuffer1+i, sizeof(Complex<T>)*(1llu << swapBufferSizeLog2));
            if (number_of_gpu > 1) hipMalloc(swapBuffer2+i, sizeof(Complex<T>)*(1llu << swapBufferSizeLog2));
            gpuc[i] = createGPUQuantumCircuitAsync(mycircuit);
            //enabling direct inter device kernel communications
            for (int j = 0; j < number_of_gpu; j++){
                if (i == j) continue;
                hipDeviceEnablePeerAccess(j, 0);
            }
        }
        for (int i = 0; i < number_of_gpu; i++){
            hipDeviceSynchronize();
        }
    }
    void execute(bool displaytime = false){// initialization and end will take care of repermuting good values
        auto t1 = high_resolution_clock::now();
        initialize();
        auto t2 = high_resolution_clock::now();

        int groupid = 0;
        for (const auto& instr: instructions){
            if (instr.first == 0){
                swapCommand(instr.second);
            } else if (instr.first == 1){
                executeCommand(groupid);
                groupid++;
            }
        }
        auto t3 = high_resolution_clock::now();

        if (displaytime){
            duration<double, std::milli> ms_double_init = t2 - t1;
            duration<double, std::milli> ms_double_compute = t3 - t2;
            
            cout << "Initialization time : " << ms_double_init.count() << " ms" << endl;
            cout << "Computation time : " << ms_double_compute.count() << " ms" << endl;
        }

        Complex<T> res0;
        hipMemcpyDtoH((&res0), (hipDeviceptr_t)gpu_qbits_states[0], sizeof(Complex<T>));
        res0.print();
        cout << endl;
    }
    ~Simulator(){
        for (int i = 0; i < number_of_gpu; i++){
            if (number_of_gpu > 1) free(cswapBuffer[i]);
            hipSetDevice(i);
            hipFree(gpu_qbits_states[i]);
            if (number_of_gpu > 1){
                hipFree(swapBuffer1[i]);
                hipFree(swapBuffer2[i]);
            }
            destroyGPUQuantumCircuit(gpuc[i]);
        }
        free(gpuc);
        free(gpu_qbits_states);
        if (number_of_gpu > 1) free(swapBuffer1);
        if (number_of_gpu > 1) free(swapBuffer2);
        if (number_of_gpu > 1) free(cswapBuffer);
    }
private:
    void initialize(){
        size_t indexfor1 = 0; //start at all 0 state. This value works no matter the permutation. But should only be put on gpu 0;
        for (int i = 0; i < number_of_gpu; i++){
            hipSetDevice(i);
            int threadnumber = min(1024llu, (1llu << (nqbits - number_of_gpu_log2)));
            int blocknumber = min((1llu << 20), (1llu << (nqbits - number_of_gpu_log2))/threadnumber);
            if (i == 0){
                initialize_state<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_states[i], indexfor1);
            } else {
                initialize_state0<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_states[i]);
            }
        }
        for (int i = 0; i < number_of_gpu; i++){
            hipSetDevice(i);
            hipDeviceSynchronize();
        }
    }
    void swapqbitDirectAccess(int q1, int q2){ //q1 local, q2 global (local/local is a swap gate, global/global is a gpu permutation but useless here)
        q2 -= nqbits - number_of_gpu_log2;
        size_t mask = (1llu << q2) - 1;
        size_t mask2 = (1llu << (number_of_gpu_log2 - 1)) - 1 - mask;
        for (int i = 0; i < number_of_gpu/2; i++){
            size_t baseIndex = (i&mask) + ((i&mask2) << 1);
            size_t otherIndex = baseIndex + (1llu << q2);
            //we will ask gpus to swap
            int threadnumber = min(1024llu, (1llu << (nqbits - number_of_gpu_log2)));
            int blocknumber = min((1llu << 20), (1llu << (nqbits - number_of_gpu_log2))/threadnumber);
            int work_per_thread = max(1llu, (1llu << (nqbits - number_of_gpu_log2))/threadnumber/blocknumber);
            if (blocknumber == 1) {threadnumber /= 2;} else {blocknumber /= 2;}
            hipSetDevice(baseIndex);
            swapqbitKernelDirectAcess<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, gpu_qbits_states[baseIndex], gpu_qbits_states[otherIndex], 0, work_per_thread);
            hipSetDevice(otherIndex);
            swapqbitKernelDirectAcess<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, gpu_qbits_states[baseIndex], gpu_qbits_states[otherIndex], (1llu << (nqbits - number_of_gpu_log2))/work_per_thread/2, work_per_thread);
        }
        for (int i = 0; i < number_of_gpu; i++){
            hipDeviceSynchronize();
        }
    }
    void swapqbitBufferSwap(int q1, int q2){
        q2 -= nqbits - number_of_gpu_log2;
        size_t data_to_transfer = (1llu << (nqbits - number_of_gpu_log2 - 1));
        size_t chunk_size = min((1llu << swapBufferSizeLog2), data_to_transfer);
        size_t mask = (1llu << q2) - 1;
        size_t mask2 = (1llu << (number_of_gpu_log2 - 1)) - 1 - mask;
        for (size_t current = 0; current < data_to_transfer; current += chunk_size){
            for (int i = 0; i < number_of_gpu/2; i++){
                //swapBuffer1 will be for sending and 2 for receiving
                size_t baseIndex = (i&mask) + ((i&mask2) << 1);
                size_t otherIndex = baseIndex + (1llu << q2);
                //we will ask gpus to swap
                int threadnumber = min(1024llu, (chunk_size));
                int blocknumber = min((1llu << 12), (chunk_size)/threadnumber);
                int work_per_thread = max(1llu, chunk_size/threadnumber/blocknumber);
                hipSetDevice(baseIndex);
                swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 1llu, gpu_qbits_states[baseIndex], swapBuffer1[baseIndex], current, work_per_thread);
                hipMemcpyPeerAsync(swapBuffer2[otherIndex], otherIndex, swapBuffer1[baseIndex], baseIndex, sizeof(Complex<T>)*chunk_size, 0);
                hipSetDevice(otherIndex);
                swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 0, gpu_qbits_states[otherIndex], swapBuffer1[otherIndex], current, work_per_thread);
                hipMemcpyPeerAsync(swapBuffer2[baseIndex], baseIndex, swapBuffer1[otherIndex], otherIndex, sizeof(Complex<T>)*chunk_size, 0);
            }
            for (int i = 0; i < number_of_gpu; i++){
                hipDeviceSynchronize();
            }
            for (int i = 0; i < number_of_gpu/2; i++){
                //swapBuffer1 will be for sending and 2 for receiving
                size_t baseIndex = (i&mask) + ((i&mask2) << 1);
                size_t otherIndex = baseIndex + (1llu << q2);
                //we will ask gpus to swap
                int threadnumber = min(1024llu, (chunk_size));
                int blocknumber = min((1llu << 12), (chunk_size)/threadnumber);
                int work_per_thread = max(1llu, chunk_size/threadnumber/blocknumber);
                hipSetDevice(baseIndex);
                swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 1llu, gpu_qbits_states[baseIndex], swapBuffer2[baseIndex], current, work_per_thread);
                hipSetDevice(otherIndex);
                swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 0, gpu_qbits_states[otherIndex], swapBuffer2[otherIndex], current, work_per_thread);
            }
            for (int i = 0; i < number_of_gpu; i++){
                hipDeviceSynchronize();
            }
        }
    }
    void swapqbitIndirectBufferSwap(int q1, int q2){
        q2 -= nqbits - number_of_gpu_log2;
        size_t data_to_transfer = (1llu << (nqbits - number_of_gpu_log2 - 1));
        size_t chunk_size = min((1llu << swapBufferSizeLog2), data_to_transfer);
        size_t mask = (1llu << q2) - 1;
        size_t mask2 = (1llu << (number_of_gpu_log2 - 1)) - 1 - mask;
        for (size_t current = 0; current < data_to_transfer; current += chunk_size){
            for (int i = 0; i < number_of_gpu/2; i++){
                //swapBuffer1 will be for sending and 2 for receiving
                size_t baseIndex = (i&mask) + ((i&mask2) << 1);
                size_t otherIndex = baseIndex + (1llu << q2);
                //we will ask gpus to swap
                int threadnumber = min(1024llu, (chunk_size));
                int blocknumber = min((1llu << 12), (chunk_size)/threadnumber);
                int work_per_thread = max(1llu, chunk_size/threadnumber/blocknumber);
                hipSetDevice(baseIndex);
                swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 1llu, gpu_qbits_states[baseIndex], swapBuffer1[baseIndex], current, work_per_thread);
                hipMemcpyDtoHAsync(cswapBuffer[baseIndex], (hipDeviceptr_t)swapBuffer1[baseIndex], sizeof(Complex<T>)*chunk_size, 0);
                hipSetDevice(otherIndex);
                swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 0, gpu_qbits_states[otherIndex], swapBuffer1[otherIndex], current, work_per_thread);
                hipMemcpyDtoHAsync(cswapBuffer[otherIndex], (hipDeviceptr_t)swapBuffer1[otherIndex], sizeof(Complex<T>)*chunk_size, 0);
            }
            for (int i = 0; i < number_of_gpu; i++){
                hipDeviceSynchronize();
            }
            for (int i = 0; i < number_of_gpu/2; i++){
                //swapBuffer1 will be for sending and 2 for receiving
                size_t baseIndex = (i&mask) + ((i&mask2) << 1);
                size_t otherIndex = baseIndex + (1llu << q2);
                //we will ask gpus to swap
                int threadnumber = min(1024llu, (chunk_size));
                int blocknumber = min((1llu << 12), (chunk_size)/threadnumber);
                int work_per_thread = max(1llu, chunk_size/threadnumber/blocknumber);
                hipSetDevice(baseIndex);
                hipMemcpyHtoDAsync((hipDeviceptr_t)swapBuffer2[baseIndex], cswapBuffer[otherIndex], sizeof(Complex<T>)*chunk_size, 0);
                swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 1llu, gpu_qbits_states[baseIndex], swapBuffer2[baseIndex], current, work_per_thread);
                hipSetDevice(otherIndex);
                hipMemcpyHtoDAsync((hipDeviceptr_t)swapBuffer2[otherIndex], cswapBuffer[baseIndex], sizeof(Complex<T>)*chunk_size, 0);
                swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 0, gpu_qbits_states[otherIndex], swapBuffer2[otherIndex], current, work_per_thread);
            }
            for (int i = 0; i < number_of_gpu; i++){
                hipDeviceSynchronize();
            }
        }
    }
    void swapCommand(vector<int> pairset){
        for (int i = 0; i < pairset.size()/2; i++){
            int q1 = pairset[2*i];
            int q2 = pairset[2*i+1];
            if (q2 < q1) swap(q1, q2);
            swapqbitIndirectBufferSwap(q1, q2);
        }
    }
    void executeCommand(int groupind){
        set<int> newqbits = groups[groupind].second;
        //we will add some qbits to make use of a block. Ideally, we should have at least 10
        for (int l = 0; l < (nqbits - number_of_gpu_log2); l++){
            if (newqbits.size() >= 10 || newqbits.size() == (nqbits - number_of_gpu_log2)) break;
            if (newqbits.find(l) != newqbits.end()) continue;
            newqbits.insert(l);
        }
        int i,j;
        vector<int> qbits(newqbits.begin(), newqbits.end());
        if (groupind == 0){
            i = 0;
        } else {
            i = groups[groupind-1].first;
        }
        j = groups[groupind].first;

        int** groupqbitsgpu = (int**)malloc(sizeof(int*)*number_of_gpu);
        for (int m = 0; m < number_of_gpu; m++){
            hipSetDevice(m);
            hipMalloc(groupqbitsgpu+m, sizeof(int)*qbits.size());
            hipMemcpyHtoDAsync((hipDeviceptr_t)groupqbitsgpu[m], qbits.data(), sizeof(int)*qbits.size(), 0);
        }
    
        for (int m = 0; m < number_of_gpu; m++){
            hipSetDevice(m);
            hipDeviceProp_t devattr;
            int device;
            hipGetDevice(&device);
	        hipGetDeviceProperties(&devattr, device);
            size_t totalshared_block = devattr.sharedMemPerBlock;
            int threadnumber = min(1024llu, (1llu << (qbits.size())));
            int blocknumber = min((1llu << 20), (1llu << ((nqbits - number_of_gpu_log2) - qbits.size())));
            executeGroupKernelSharedState<<<dim3(blocknumber), dim3(threadnumber), totalshared_block, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_states[m], qbits.size(), groupqbitsgpu[m], gpuc[m].gates+i, j-i, totalshared_block - sizeof(Complex<T>)*(1llu << qbits.size()));
        }

        for (int m = 0; m < number_of_gpu; m++){
            hipSetDevice(m);
            hipDeviceSynchronize();
            hipFree(groupqbitsgpu[m]);
        }
        free(groupqbitsgpu);
    }
};