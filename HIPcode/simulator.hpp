#ifndef SIMULATORDONE
#define SIMULATORDONE

#include "HIPpreprocessor.hpp"
#include "../backendagnosticcode/QuantumCircuit.hpp"
#include "GPUQuantumCircuit.hpp"
#include "../backendagnosticcode/basic_host_types.hpp"
#include "../backendagnosticcode/Circuit.hpp"

namespace Kate {

__global__ void printKernel(Complex* mem){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    printf("at %zu there is %f\n", tid, mem[tid].a);
}

__global__ void initialize_state(int nqbits, Complex* memory, int indexfor1){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    int work_per_thread = (1llu << nqbits)/blockDim.x/gridDim.x;

    for (size_t i = tid*work_per_thread; i < (tid+1)*work_per_thread; i++){
        memory[i] = (i == indexfor1)? 1 : 0;
    }
}

__global__ void initialize_probastate(int nqbits, Complex* memory, Complex* qbitsangles, Complex offset){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    int work_per_thread = (1llu << nqbits)/blockDim.x/gridDim.x;

    Complex temp;
    for (size_t i = tid*work_per_thread; i < (tid+1)*work_per_thread; i++){
        temp = offset;
        for (int j = 0; j < nqbits; j++){
            temp *= qbitsangles[((i >> j)%2)*nqbits + j];
        }
        memory[i] = temp;
    }
}

__global__ void measureKernel(int nqbits, Complex* qbitsstate, Complex* allresultsend){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    //int work_per_block = (1llu << nqbits)/gridDim.x;
    int work_per_thread = (1llu << nqbits)/blockDim.x/gridDim.x;
    //we need 2*nqbits complex to save our results
    allresultsend += (2*nqbits)*tid;
    Complex allresults[64]; //all 0 then all 1
    for (int i = 0; i < 2*nqbits; i++){
        allresults[i].a = 0.;
        allresults[i].b = 0.;
    }

    for (int i = tid*work_per_thread; i < (tid+1)*work_per_thread; i++){
        for (int qbit = 0; qbit < nqbits; qbit++){
            allresults[qbit*2 + ((i >> qbit)%2)] += qbitsstate[i];
        }
    }

    for (int i = 0; i < 2*nqbits; i++){
        allresultsend[i] = allresults[i];
    }
}

__global__ void measureKernelqbit(int nqbits, Complex* qbitsstate, Complex* allresultsend, int qbit){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    //int work_per_block = (1llu << nqbits)/gridDim.x;
    int work_per_thread = (1llu << nqbits)/blockDim.x/gridDim.x;
    
    extern __shared__ Complex res[];
    Complex* myres = res + threadIdx.x*2;

    for (int i = tid*work_per_thread; i < (tid+1)*work_per_thread; i++){
        myres[(i >> qbit)%2] += qbitsstate[i];
    }

    __syncthreads();
    //pointer jumping
    int i = 1;
    while (i < blockDim.x){
        if (threadIdx.x+i < blockDim.x && threadIdx.x%(2*i) == 0){
            myres[0] += myres[2*i];
            myres[1] += myres[2*i+1];
        }
        i *= 2;
        __syncthreads();
    }

    if (threadIdx.x == 0){
        allresultsend[blockIdx.x*2] = myres[0];
        allresultsend[blockIdx.x*2+1] = myres[1];
    }
}

__global__ void initialize_state0(int nqbits, Complex* memory){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    int work_per_thread = (1llu << nqbits)/blockDim.x/gridDim.x;

    for (size_t i = tid*work_per_thread; i < (tid+1)*work_per_thread; i++){
        memory[i] = 0;
    }
}

__global__ void swapqbitKernelDirectAcess(int nqbits, int localq, Complex* memory0, Complex* memory1, int baseindex, int values_per_thread){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;

    size_t mask = (1llu << localq) - 1;
    size_t mask2 = (1llu << (nqbits - 1)) - 1 - mask;

    for (size_t i = tid*values_per_thread; i < (tid+1)*(values_per_thread); i++){
        size_t baseIndex = ((i+baseindex)&mask) + (((i+baseindex)&mask2) << 1);

        Complex temp = memory1[baseIndex]; //we want the 0 of device which has the global 1;
        memory1[baseIndex] = memory0[baseIndex + (1llu << localq)]; //then we paste the 1 of the device which has the global 0
        memory0[baseIndex + (1llu << localq)] = temp;
    }
}

__global__ void swapqbitKernelIndirectAccessEXTRACT(int nqbits, int localq, size_t qbitvalue, Complex* mymemory, Complex* buffer, int baseindex, int values_per_thread){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;

    size_t mask = (1llu << localq) - 1;
    size_t mask2 = (1llu << (nqbits - 1)) - 1 - mask;

    for (size_t i = tid*values_per_thread; i < (tid+1)*(values_per_thread); i++){
        size_t value = ((i+baseindex)&mask) + (((i+baseindex)&mask2) << 1);
        
        buffer[i] = mymemory[value + ((qbitvalue) << localq)];
    }
}

__global__ void swapqbitKernelIndirectAccessIMPORT(int nqbits, int localq, size_t qbitvalue, Complex* mymemory, Complex* buffer, int baseindex, int values_per_thread){
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;

    size_t mask = (1llu << localq) - 1;
    size_t mask2 = (1llu << (nqbits - 1)) - 1 - mask;

    for (size_t i = tid*values_per_thread; i < (tid+1)*(values_per_thread); i++){
        size_t value = ((i+baseindex)&mask) + (((i+baseindex)&mask2) << 1);
        
        mymemory[value + ((qbitvalue) << localq)] = buffer[i];
    }
}

__global__ void executeGroupKernelSharedState(int nqbits, Complex* qbitsstate, int groupnqbits, int* groupqbits, GPUGate* gates, int gatenumber, int sharedMemMatrixSize){
    int bit_to_groupbitnumber[64];
    for (int i = 0; i < groupnqbits; i++){
        bit_to_groupbitnumber[groupqbits[i]] = i;
    }

    extern __shared__ Complex qbitsstateANDmatrixstate[]; //must be of size sizeof(T)*2**nbqbits + sharedMemMatrixSize
    Complex* qbitsstateshared = qbitsstateANDmatrixstate; //size 2**(groupnqbits)
    Complex* matrixsharedstorage = qbitsstateANDmatrixstate + (1llu << groupnqbits); //size sharedMemMatrixSize

    size_t mask_group[64];
    size_t cumulative = 0;
    for (int i = 0; i < groupnqbits; i++){
        mask_group[i] = (1llu << (groupqbits[i] - i)) - 1 - cumulative;
        cumulative += mask_group[i];
    }
    mask_group[groupnqbits] = (1llu << (nqbits - groupnqbits)) - 1 - cumulative;

    size_t groupnumber = (1llu << (nqbits - groupnqbits));
    size_t groupsperblock = groupnumber/gridDim.x;
    for (size_t groupel = blockIdx.x*groupsperblock; groupel < (blockIdx.x +1)*groupsperblock; groupel++){
        size_t initline = threadIdx.x;


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

    }

    //__syncthreads();
}

class Simulator{
public:
    //copy from quantum circuit but with the gpu version
    std::vector<std::pair<int, std::set<int>>> groups; //second is current qbit set, first is when to go to next group
    std::vector<int> initial_permutation;
    std::vector<int> final_inverse_permutation;
    std::vector<std::pair<int, std::vector<int>>> instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
    std::vector<Gate> gate_set_ordered;
    int nqbits = 0;

    int number_of_gpu;
    int number_of_gpu_log2;
    Complex** gpu_qbits_states;
    GPUQuantumCircuit* gpuc; //one for each device

    Complex** swapBuffer1 = NULL;
    Complex** swapBuffer2 = NULL;
    Complex** cswapBuffer = NULL;
    int swapBufferSizeLog2;

    bool cpumode = false;
    CPUSimulator cpusim;

    Simulator(QuantumCircuit mycircuit, int number_of_gpu, int swapBufferSizeLog2 = 24){
        int count;
        if (hipGetDeviceCount(&count) != 0){
            std::cout << "Warning: Simulator Could not detect a GPU: Fallback on CPU" << std::endl;
            cpumode = true;
            CPUSimulator cpusimtest(mycircuit, number_of_gpu, swapBufferSizeLog2);
            cpusim = cpusimtest;
            return;
        } else if (count == 0){
            std::cout << "Warning: Simulator detected no GPU: Fallback on CPU" << std::endl;
            cpumode = true;
            cpusim = CPUSimulator(mycircuit, number_of_gpu, swapBufferSizeLog2);
            return;
        } else if (count < number_of_gpu){
            std::cout << "Warning: You attempted to use more GPUs than I am able to detect I will use the maximum number of GPU possible but expect errors since you might have compiled the circuit for the wrong number of global qubit" << std::endl;
            number_of_gpu = (1llu << (int)log2(count));
        }

        if (mycircuit.instructions.size() == 0){
            std::cout << "warning: the simulator has been input a circuit that is not compiled. I will compile it naively now" << std::endl;
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
        gpu_qbits_states = (Complex**)malloc(sizeof(Complex*)*number_of_gpu);
        if (number_of_gpu > 1) swapBuffer1 = (Complex**)malloc(sizeof(Complex*)*number_of_gpu);
        if (number_of_gpu > 1) swapBuffer2 = (Complex**)malloc(sizeof(Complex*)*number_of_gpu);
        if (number_of_gpu > 1) cswapBuffer = (Complex**)malloc(sizeof(Complex*)*number_of_gpu);
        
        this->swapBufferSizeLog2 = swapBufferSizeLog2;
        gpuc = (GPUQuantumCircuit*)malloc(sizeof(GPUQuantumCircuit)*number_of_gpu);
        for (int i = 0; i < number_of_gpu; i++){
            if (number_of_gpu > 1) cswapBuffer[i] = (Complex*)malloc(sizeof(Complex)*(1llu << swapBufferSizeLog2));
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipMalloc(gpu_qbits_states+i, sizeof(Complex)*(1llu << (nqbits - number_of_gpu_log2))));
            if (number_of_gpu > 1) {GPU_CHECK(hipMalloc(swapBuffer1+i, sizeof(Complex)*(1llu << swapBufferSizeLog2)))};
            if (number_of_gpu > 1) {GPU_CHECK(hipMalloc(swapBuffer2+i, sizeof(Complex)*(1llu << swapBufferSizeLog2)))};
            gpuc[i] = createGPUQuantumCircuitAsync(mycircuit);
            //enabling direct inter device kernel communications
            for (int j = 0; j < number_of_gpu; j++){
                if (i == j) continue;
                errhip = hipDeviceEnablePeerAccess(j, 0);
                if ((int)errhip != 704){
                    GPU_CHECK(errhip);
                }
            }
        }
        for (int i = 0; i < number_of_gpu; i++){
            GPU_CHECK(hipDeviceSynchronize());
        }
    }
    proba_state execute(bool displaytime = false){// initialization and end will take care of repermuting good values
        if (cpumode) return cpusim.execute(displaytime);

        double inittime = -1;
        double measuretime = -1;
        double groupcomputetime = 0;
        int groupnumber = 0;
        double swaptime = 0;
        int swapnumber = 0;
        
        auto t1 = high_resolution_clock::now();
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;

        t1 = high_resolution_clock::now();
        initialize();
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        inittime = ms_double.count();

        int groupid = 0;
        for (const auto& instr: instructions){
            t1 = high_resolution_clock::now();
            if (instr.first == 0){
                swapCommand(instr.second);
            } else if (instr.first == 1){
                executeCommand(groupid);
                groupid++;
            }
            t2 = high_resolution_clock::now();
            ms_double = t2 - t1;
            if (instr.first == 0){
                swaptime += ms_double.count();
                swapnumber++;
            } else {
                groupcomputetime += ms_double.count();
                groupnumber++;
            }
        }

        t1 = high_resolution_clock::now();
        auto res = measurement();
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        measuretime = ms_double.count();

        if (displaytime){
            std::cout << "Initialization/Measurement : " << inittime << " / " << measuretime << " ms" << std::endl;
            std::cout << "Total computing time : " << swaptime+groupcomputetime << " ms with swap% : " << 100*swaptime/(swaptime+groupcomputetime) << "%" << std::endl;
            std::cout << "Average swap time : " << swaptime/swapnumber << " ms and average group time : " << groupcomputetime/groupnumber << " ms" << std::endl;
        }
        return res;
    }
    proba_state execute(proba_state& in, bool displaytime = false){// initialization and end will take care of repermuting good values
        if (cpumode) return cpusim.execute(displaytime);

        double inittime = -1;
        double measuretime = -1;
        double groupcomputetime = 0;
        int groupnumber = 0;
        double swaptime = 0;
        int swapnumber = 0;
        
        auto t1 = high_resolution_clock::now();
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;

        t1 = high_resolution_clock::now();
        initialize(in);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        inittime = ms_double.count();

        int groupid = 0;
        for (const auto& instr: instructions){
            t1 = high_resolution_clock::now();
            if (instr.first == 0){
                swapCommand(instr.second);
            } else if (instr.first == 1){
                executeCommand(groupid);
                groupid++;
            }
            t2 = high_resolution_clock::now();
            ms_double = t2 - t1;
            if (instr.first == 0){
                swaptime += ms_double.count();
                swapnumber++;
            } else {
                groupcomputetime += ms_double.count();
                groupnumber++;
            }
        }

        t1 = high_resolution_clock::now();
        auto res = measurement();
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        measuretime = ms_double.count();

        if (displaytime){
            std::cout << "Initialization/Measurement : " << inittime << " / " << measuretime << " ms" << std::endl;
            std::cout << "Total computing time : " << swaptime+groupcomputetime << " ms with swap% : " << 100*swaptime/(swaptime+groupcomputetime) << "%" << std::endl;
            std::cout << "Average swap time : " << swaptime/swapnumber << " ms and average group time : " << groupcomputetime/groupnumber << " ms" << std::endl;
        }
        return res;
    }
    ~Simulator(){
        if (cpumode) return;

        for (int i = 0; i < number_of_gpu; i++){
            if (number_of_gpu > 1) free(cswapBuffer[i]);
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipFree(gpu_qbits_states[i]));
            if (number_of_gpu > 1){
                GPU_CHECK(hipFree(swapBuffer1[i]));
                GPU_CHECK(hipFree(swapBuffer2[i]));
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
            GPU_CHECK(hipSetDevice(i));
            int threadnumber = min(1024llu, (1llu << (nqbits - number_of_gpu_log2)));
            int blocknumber = min((1llu << 20), (1llu << (nqbits - number_of_gpu_log2))/threadnumber);
            if (i == 0){
                initialize_state<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_states[i], indexfor1);
            } else {
                initialize_state0<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_states[i]);
            }
        }
        for (int i = 0; i < number_of_gpu; i++){
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipDeviceSynchronize());
        }
    }
    void initialize(proba_state& state_input){
        if (state_input.val.size() != nqbits){
            std::cout << "wrong input proba_state_size_input, defaulting to no input" << std::endl;
            initialize();
            return;
        }
        std::vector<Complex> allstates(2*nqbits);
        std::vector<Complex> gpustates(2*(nqbits-number_of_gpu_log2));
        for (int i = 0; i < nqbits; i++){
            Complex val0, val1;
            val0 = Complex(cos((state_input.val[i].first)*PI/2), 0);
            val1 = Complex(cos((state_input.val[i].second)), sin((state_input.val[i].second)))*sin((state_input.val[i].first)*PI/2);
            allstates[initial_permutation[i]] = val0;
            allstates[initial_permutation[i] + nqbits] = val1;
        }
        for (int i = 0; i < nqbits - number_of_gpu_log2; i++){
            gpustates[i] = allstates[i];
            gpustates[i+nqbits-number_of_gpu_log2] = allstates[i+nqbits];
        }
        Complex** anglesinter_d = (Complex**)malloc(sizeof(Complex*)*number_of_gpu);
        Complex offset;
        for (int i = 0; i < number_of_gpu; i++){
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipMalloc(anglesinter_d+i, sizeof(Complex)*2*(nqbits - number_of_gpu_log2)));
            GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)anglesinter_d[i], gpustates.data(), sizeof(Complex)*2*(nqbits - number_of_gpu_log2), 0));
            int threadnumber = min(1024llu, (1llu << (nqbits - number_of_gpu_log2)));
            int blocknumber = min((1llu << 20), (1llu << (nqbits - number_of_gpu_log2))/threadnumber);
            offset = Complex(1, 0);
            for (int j = 0; j < number_of_gpu_log2; j++){
                offset = offset * (allstates[((i >> j)%2)*nqbits + j + nqbits-number_of_gpu_log2]);
            }
            initialize_probastate<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>(nqbits-number_of_gpu_log2, gpu_qbits_states[i], anglesinter_d[i], offset);
        }
        for (int i = 0; i < number_of_gpu; i++){
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipDeviceSynchronize());
            GPU_CHECK(hipFree(anglesinter_d[i]));   
        }
    }
    proba_state measurement(){
        int threadnumber = min(1024llu, (1llu << (nqbits - number_of_gpu_log2)));
        int blocknumber = min((1llu << 5), (1llu << (nqbits - number_of_gpu_log2))/threadnumber);
        Complex** measureintermediate_d = (Complex**)malloc(sizeof(Complex*)*number_of_gpu);
        Complex* measureintermediate = (Complex*)malloc(sizeof(Complex)*2*(nqbits-number_of_gpu_log2)*threadnumber*blocknumber*number_of_gpu);
        Complex* measure = (Complex*)malloc(sizeof(Complex)*2*nqbits);
        Complex temp;
        
        for (int i = 0; i < number_of_gpu; i++){
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipMalloc(measureintermediate_d+i, sizeof(Complex)*(threadnumber*blocknumber*2*(nqbits-number_of_gpu_log2))));
            measureKernel<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits-number_of_gpu_log2), gpu_qbits_states[i], measureintermediate_d[i]);
            GPU_CHECK(hipMemcpyDtoHAsync(measureintermediate+(2*(nqbits-number_of_gpu_log2)*threadnumber*blocknumber*i), (hipDeviceptr_t)measureintermediate_d[i], sizeof(Complex)*2*(nqbits-number_of_gpu_log2)*threadnumber*blocknumber, 0));
        }

        for (int i = 0; i < 2*nqbits; i++){
            measure[i] = 0;
        }

        for (int i = 0; i < number_of_gpu; i++){
            temp = 0;
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipDeviceSynchronize());
            GPU_CHECK(hipFree(measureintermediate_d[i]));
            for (int j = 0; j < threadnumber*blocknumber; j++){
                for (int k = 0; k < 2*(nqbits-number_of_gpu_log2); k++){
                    measure[k] += measureintermediate[i*threadnumber*blocknumber*2*(nqbits-number_of_gpu_log2) + j*2*(nqbits-number_of_gpu_log2) + k];
                }
                temp += measureintermediate[i*threadnumber*blocknumber*2*(nqbits-number_of_gpu_log2) + j*2*(nqbits-number_of_gpu_log2) + 0];
                temp += measureintermediate[i*threadnumber*blocknumber*2*(nqbits-number_of_gpu_log2) + j*2*(nqbits-number_of_gpu_log2) + 1];
            }
            for (int j = 0; j < number_of_gpu_log2; j++){
                measure[((i >> j)%2) + 2*(j+(nqbits - number_of_gpu_log2))] += temp;
            }
        }

        //now we just need to get the spin
        std::vector<std::pair<double,  double>> res(nqbits);
        for (int i = 0; i < nqbits; i++){
            Complex val0 = measure[2*i]/pow(SQRT2, (double)nqbits);
            Complex val1 = measure[2*i+1]/pow(SQRT2, (double)nqbits);
            double teta, phi;
            if (val0.norm() < 0.0000000000001){
                teta = 1;
                phi = 0;
            } else if (val1.norm() < 0.00000000000001) {
                teta = 0;
                phi = 0;
            } else {
                teta = atan((val1.norm())/(val0.norm()))/(PI/2);
                phi = (val1/val0).angle();
            }
            res[final_inverse_permutation[i]] = std::make_pair(teta, phi);
        }

        free(measureintermediate);
        free(measureintermediate_d);
        free(measure);

        return proba_state(res);
    }
    proba_state measurement3(){ //full multi threaded cpu version
        int localqbits = nqbits - number_of_gpu_log2;
        int usable_threads = min((int)THREADNUMBER, (1 << localqbits));
        Complex* buffer1 = (Complex*)malloc(sizeof(Complex)*(1llu << localqbits));
        Complex* buffer2 = (Complex*)malloc(sizeof(Complex)*(1llu << localqbits));
        Complex* threadres = (Complex*)malloc(sizeof(Complex)*usable_threads*localqbits*2);
        Complex* measure = (Complex*)malloc(sizeof(Complex)*localqbits*2);
        std::vector<std::thread> threads;
        Complex temp;

        for (int i = 0; i < usable_threads*2*localqbits; i++){
            threadres[i] = 0;
        }
        for (int i = 0; i < 2*localqbits; i++){
            measure[i] = 0;
        }
        
        auto threadef = [&localqbits](Complex* res, Complex* buffer, size_t i, size_t j){
            for (int i = 0; i < j; i++){
                for (int qbit = 0; qbit < localqbits; qbit++){
                    res[qbit*2 + ((i >> qbit)%2)] += buffer[i];
                }
            }
        };


        //parallel load from gpu and compute on cpu
        GPU_CHECK(hipSetDevice(0));
        GPU_CHECK(hipMemcpyDtoH(buffer1, (hipDeviceptr_t)gpu_qbits_states[0], sizeof(Complex)*(1llu << localqbits)));

        for (int i = 0; i < number_of_gpu; i++){
            threads.clear();
            if (i+1 < number_of_gpu){
                GPU_CHECK(hipSetDevice(i+1));
                GPU_CHECK(hipMemcpyDtoHAsync(buffer2, (hipDeviceptr_t)gpu_qbits_states[i+1], sizeof(Complex)*(1llu << localqbits), 0));
            }
            //let's compute for buffer 1
            size_t work_per_thread = ((1llu << localqbits)/usable_threads);
            for (int j = 0; j < usable_threads; j++){
                threads.emplace_back(threadef, threadres+2*localqbits*j, buffer1, (size_t)j*work_per_thread, (size_t)min(1llu << localqbits, (unsigned long long)(j+1)*work_per_thread));
            }

            for (auto& el: threads){
                el.join();
            }

            //now I just need to get the total measure for global qbits purpose
            temp = 0;
            for (int j = 0; j < usable_threads; j++){
                temp += threadres[j*localqbits*2];
                temp += threadres[j*localqbits*2+1]; //only need qbit 0 at value 0 and 1 for that
            }

            for (int j = 0; j < number_of_gpu_log2; j++){
                measure[((i >> j)%2) + 2*(j+localqbits)] += temp;
            }

            if (i+1 < number_of_gpu) {GPU_CHECK(hipDeviceSynchronize())};

            std::swap(buffer1, buffer2);
        }
        //now we can add the results of every threads to end all localqbits
        for (int i = 0; i < localqbits; i++){
            for (int j = 0; j < usable_threads; j++){
                measure[2*i] += threadres[j*localqbits*2 + 2*i];
                measure[2*i+1] += threadres[j*localqbits*2 + 2*i+1];
            }
        }

        //now we just need to get the spin
        std::vector<std::pair<double,  double>> res(nqbits);
        for (int i = 0; i < nqbits; i++){
            Complex val0 = measure[2*i]/pow(SQRT2, (double)nqbits);
            Complex val1 = measure[2*i+1]/pow(SQRT2, (double)nqbits);
            double teta, phi;
            if (val0.norm() < 0.0000000000001){
                teta = 1;
                phi = 0;
            } else if (val1.norm() < 0.00000000000001) {
                teta = 0;
                phi = 0;
            } else {
                teta = atan((val1.norm())/(val0.norm()))/(PI/2);
                phi = (val1/val0).angle();
            }
            res[final_inverse_permutation[i]] = std::make_pair(teta, phi);
        }

        free(buffer1);
        free(buffer2);
        free(threadres);
        free(measure);

        return proba_state(res);        
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
            if (blocknumber == 1) {threadnumber /= 2;} else {blocknumber /= 2;} //half data

            if (blocknumber == 1) {threadnumber /= 2;} else {blocknumber /= 2;}//for 2nd gpu work
            GPU_CHECK(hipSetDevice(baseIndex));
            swapqbitKernelDirectAcess<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, gpu_qbits_states[baseIndex], gpu_qbits_states[otherIndex], 0, work_per_thread);
            GPU_CHECK(hipSetDevice(otherIndex));
            swapqbitKernelDirectAcess<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, gpu_qbits_states[baseIndex], gpu_qbits_states[otherIndex], (1llu << (nqbits - number_of_gpu_log2))/4, work_per_thread);
        }
        for (int i = 0; i < number_of_gpu; i++){
            GPU_CHECK(hipDeviceSynchronize());
        }
    }
    void swapqbitBufferSwap(int q1, int q2){
        q2 -= nqbits - number_of_gpu_log2;
        size_t data_to_transfer = (1llu << (nqbits - number_of_gpu_log2 - 1));
        size_t chunk_size = min((1llu << swapBufferSizeLog2), (unsigned long long)data_to_transfer);
        size_t mask = (1llu << q2) - 1;
        size_t mask2 = (1llu << (number_of_gpu_log2 - 1)) - 1 - mask;

        for (size_t current = 0; current < data_to_transfer; current += chunk_size){
            for (int i = 0; i < number_of_gpu/2; i++){
                //swapBuffer1 will be for sending and 2 for receiving
                size_t baseIndex = (i&mask) + ((i&mask2) << 1);
                size_t otherIndex = baseIndex + (1llu << q2);
                //we will ask gpus to swap
                int threadnumber = min(1024llu, (unsigned long long)(chunk_size));
                int blocknumber = min((1llu << 12), (unsigned long long)(chunk_size)/threadnumber);
                int work_per_thread = max(1llu, (unsigned long long)chunk_size/threadnumber/blocknumber);
                GPU_CHECK(hipSetDevice(baseIndex));
                swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 1llu, gpu_qbits_states[baseIndex], swapBuffer1[baseIndex], current, work_per_thread);
                GPU_CHECK(hipMemcpyPeer(swapBuffer2[otherIndex], otherIndex, swapBuffer1[baseIndex], baseIndex, sizeof(Complex)*chunk_size));
                GPU_CHECK(hipSetDevice(otherIndex));
                swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 0, gpu_qbits_states[otherIndex], swapBuffer1[otherIndex], current, work_per_thread);
                GPU_CHECK(hipMemcpyPeerAsync(swapBuffer2[baseIndex], baseIndex, swapBuffer1[otherIndex], otherIndex, sizeof(Complex)*chunk_size, 0));
            }
            for (int i = 0; i < number_of_gpu; i++){
                GPU_CHECK(hipDeviceSynchronize());
            }
            for (int i = 0; i < number_of_gpu/2; i++){
                //swapBuffer1 will be for sending and 2 for receiving
                size_t baseIndex = (i&mask) + ((i&mask2) << 1);
                size_t otherIndex = baseIndex + (1llu << q2);
                //we will ask gpus to swap
                int threadnumber = min(1024llu, (unsigned long long)(chunk_size));
                int blocknumber = min((1llu << 12), (unsigned long long)(chunk_size)/threadnumber);
                int work_per_thread = max(1llu, (unsigned long long)chunk_size/threadnumber/blocknumber);
                GPU_CHECK(hipSetDevice(baseIndex));
                swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 1llu, gpu_qbits_states[baseIndex], swapBuffer2[baseIndex], current, work_per_thread);
                GPU_CHECK(hipSetDevice(otherIndex));
                swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 0, gpu_qbits_states[otherIndex], swapBuffer2[otherIndex], current, work_per_thread);
            }
            for (int i = 0; i < number_of_gpu; i++){
                GPU_CHECK(hipDeviceSynchronize());
            }
        }
    }
    void swapqbitIndirectBufferSwap(int q1, int q2){
        q2 -= nqbits - number_of_gpu_log2;
        size_t data_to_transfer = (1llu << (nqbits - number_of_gpu_log2 - 1));
        size_t chunk_size = min((size_t)(1llu << swapBufferSizeLog2), data_to_transfer);
        size_t mask = (1llu << q2) - 1;
        size_t mask2 = (1llu << (number_of_gpu_log2 - 1)) - 1 - mask;
        for (size_t current = 0; current < data_to_transfer; current += chunk_size){
            for (int i = 0; i < number_of_gpu/2; i++){
                //swapBuffer1 will be for sending and 2 for receiving
                size_t baseIndex = (i&mask) + ((i&mask2) << 1);
                size_t otherIndex = baseIndex + (1llu << q2);
                //we will ask gpus to swap
                int threadnumber = min(1024llu, (unsigned long long)(chunk_size));
                int blocknumber = min((1llu << 12), (unsigned long long)(chunk_size)/threadnumber);
                int work_per_thread = max(1llu, (unsigned long long)chunk_size/threadnumber/blocknumber);
                GPU_CHECK(hipSetDevice(baseIndex));
                swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 1llu, gpu_qbits_states[baseIndex], swapBuffer1[baseIndex], current, work_per_thread);
                GPU_CHECK(hipMemcpyDtoHAsync(cswapBuffer[baseIndex], (hipDeviceptr_t)swapBuffer1[baseIndex], sizeof(Complex)*chunk_size, 0));
                GPU_CHECK(hipSetDevice(otherIndex));
                swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 0, gpu_qbits_states[otherIndex], swapBuffer1[otherIndex], current, work_per_thread);
                GPU_CHECK(hipMemcpyDtoHAsync(cswapBuffer[otherIndex], (hipDeviceptr_t)swapBuffer1[otherIndex], sizeof(Complex)*chunk_size, 0));
            }
            for (int i = 0; i < number_of_gpu; i++){
                GPU_CHECK(hipDeviceSynchronize());
            }
            for (int i = 0; i < number_of_gpu/2; i++){
                //swapBuffer1 will be for sending and 2 for receiving
                size_t baseIndex = (i&mask) + ((i&mask2) << 1);
                size_t otherIndex = baseIndex + (1llu << q2);
                //we will ask gpus to swap
                int threadnumber = min(1024llu, (unsigned long long)(chunk_size));
                int blocknumber = min((1llu << 12), (unsigned long long)(chunk_size)/threadnumber);
                int work_per_thread = max(1llu, (unsigned long long)chunk_size/threadnumber/blocknumber);
                GPU_CHECK(hipSetDevice(baseIndex));
                GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)swapBuffer2[baseIndex], cswapBuffer[otherIndex], sizeof(Complex)*chunk_size, 0));
                swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 1llu, gpu_qbits_states[baseIndex], swapBuffer2[baseIndex], current, work_per_thread);
                GPU_CHECK(hipSetDevice(otherIndex));
                GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)swapBuffer2[otherIndex], cswapBuffer[baseIndex], sizeof(Complex)*chunk_size, 0));
                swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, 0, gpu_qbits_states[otherIndex], swapBuffer2[otherIndex], current, work_per_thread);
            }
            for (int i = 0; i < number_of_gpu; i++){
                GPU_CHECK(hipDeviceSynchronize());
            }
        }
    }
    void swapCommand(std::vector<int> pairset){
        for (int i = 0; i < pairset.size()/2; i++){
            int q1 = pairset[2*i];
            int q2 = pairset[2*i+1];
            if (q2 < q1) std::swap(q1, q2);
            swapqbitDirectAccess(q1, q2);
            //swapqbitBufferSwap(q1, q2);
        }
    }
    void executeCommand(int groupind){
        std::set<int> newqbits = groups[groupind].second;
        //we will add some qbits to make use of a block. Ideally, we should have at least 10
        for (int l = 0; l < (nqbits - number_of_gpu_log2); l++){
            if (newqbits.size() >= 8 || newqbits.size() == (nqbits - number_of_gpu_log2)) break;
            if (newqbits.find(l) != newqbits.end()) continue;
            newqbits.insert(l);
        }
        int i,j;
        std::vector<int> qbits(newqbits.begin(), newqbits.end());
        if (groupind == 0){
            i = 0;
        } else {
            i = groups[groupind-1].first;
        }
        j = groups[groupind].first;

        int** groupqbitsgpu = (int**)malloc(sizeof(int*)*number_of_gpu);
        for (int m = 0; m < number_of_gpu; m++){
            GPU_CHECK(hipSetDevice(m));
            GPU_CHECK(hipMalloc(groupqbitsgpu+m, sizeof(int)*qbits.size()));
            GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)groupqbitsgpu[m], qbits.data(), sizeof(int)*qbits.size(), 0));
        }
    
        for (int m = 0; m < number_of_gpu; m++){
            GPU_CHECK(hipSetDevice(m));
            hipDeviceProp_t devattr;
            int device;
            GPU_CHECK(hipGetDevice(&device));
	        GPU_CHECK(hipGetDeviceProperties(&devattr, device));
            size_t totalshared_block = devattr.sharedMemPerBlock/4;
            int threadnumber = min(1024llu, (1llu << (qbits.size())));
            int blocknumber = min((1llu << 20), (1llu << ((nqbits - number_of_gpu_log2) - qbits.size())));
            if ((1llu << qbits.size()) > totalshared_block){
                std::cout << "too much qbits in one group for this gpu's shared memory... I cancel this group's computation" << std::endl;
                continue;
            }
            executeGroupKernelSharedState<<<dim3(blocknumber), dim3(threadnumber), totalshared_block, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_states[m], qbits.size(), groupqbitsgpu[m], gpuc[m].gates+i, j-i, totalshared_block - sizeof(Complex)*(1llu << qbits.size()));
            hipError_t kernelret = hipGetLastError();
            if (kernelret != hipSuccess){
                std::cout << "Error while trying to compute a group, the  kernel refused to launch for this reason : " << std::endl << hipGetErrorString(kernelret) << std::endl << "It could be due to a lack of shared cache, please reduce the size of each group" << std::endl;
            }
        }

        for (int m = 0; m < number_of_gpu; m++){
            GPU_CHECK(hipSetDevice(m));
            GPU_CHECK(hipDeviceSynchronize());
            GPU_CHECK(hipFree(groupqbitsgpu[m]));
        }
        free(groupqbitsgpu);
    }
};

}

#endif