#ifndef SIMULATORDONE
#define SIMULATORDONE

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
    int localqbits;
    Complex* gpu_qbits_state;
    GPUQuantumCircuit gpuc; //one for each device

    Complex* swapBuffer1 = NULL;
    Complex* swapBuffer2 = NULL;
    int swapBufferSizeLog2;

    bool cpumode = false;
    CPUSimulator cpusim;

    int rank;
    MPI_Comm comm;

    int slowqbitsnumber = 0;

    Simulator(QuantumCircuit mycircuit, MPI_Comm comm, int swapBufferSizeLog2 = 24){
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &number_of_gpu);
        this->comm = comm;

        int count;
        if (hipGetDeviceCount(&count) != 0){
            std::cout << "Warning: Simulator Could not detect a GPU: Fallback on CPU" << std::endl;
            cpumode = true;
            CPUSimulator cpusimtest(mycircuit, comm, swapBufferSizeLog2);
            cpusim = cpusimtest;
            return;
        } else if (count == 0){
            std::cout << "Warning: Simulator detected no GPU: Fallback on CPU" << std::endl;
            cpumode = true;
            cpusim = CPUSimulator(mycircuit, comm, swapBufferSizeLog2);
            return;
        } 

        slowqbitsnumber = std::max(0, (int)log2(number_of_gpu/count));
        GPU_CHECK(hipSetDevice(rank%count)) //choice of our gpuid

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
        localqbits = nqbits - number_of_gpu_log2;
        
        this->swapBufferSizeLog2 = swapBufferSizeLog2;
        
        GPU_CHECK(hipMalloc(&gpu_qbits_state, sizeof(Complex)*(1llu << (nqbits - number_of_gpu_log2))));
        if (number_of_gpu > 1) {GPU_CHECK(hipMalloc(&swapBuffer1, sizeof(Complex)*(1llu << swapBufferSizeLog2)))};
        if (number_of_gpu > 1) {GPU_CHECK(hipMalloc(&swapBuffer2, sizeof(Complex)*(1llu << swapBufferSizeLog2)))};
        gpuc = createGPUQuantumCircuitAsync(mycircuit);
        
        GPU_CHECK(hipDeviceSynchronize());
    }
    proba_state execute(bool displaytime = false){// initialization and end will take care of repermuting good values
        if (cpumode) return cpusim.execute(displaytime);

        double inittime = -1;
        double measuretime = -1;
        double groupcomputetime = 0;
        int groupnumber = 0;
        double slowswaptime = 0;
        double fastswaptime = 0;
        int fastswapnumber = 0;
        int slowswapnumber = 0;
        
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
                for (const auto& qbit: instr.second){
                    if (qbit < localqbits) continue;
                    if (qbit < nqbits-slowqbitsnumber) {
                        fastswapnumber++;
                        fastswaptime += ms_double.count();
                        continue;
                    }
                    slowswapnumber++;
                    slowswaptime += ms_double.count();

                }
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
            double swaptime = slowswaptime+fastswaptime;
            int swapnumber = slowswapnumber + fastswapnumber;
            std::cout << "Initialization/Measurement : " << inittime << " / " << measuretime << " ms" << std::endl;
            std::cout << "Total computing time : " << swaptime+groupcomputetime << " ms with swap% : " << 100*swaptime/(swaptime+groupcomputetime) << "%" << std::endl;
            std::cout << "Average swap time : " << swaptime/swapnumber << " ms and average group time : " << groupcomputetime/groupnumber << " ms" << std::endl;
            std::cout << "Number swap fast/slow : " << fastswapnumber << "/" << slowqbitsnumber << " Time swap fast/slow : " << fastswaptime << " / " << slowswaptime << " ms" << std::endl;
        }
        return res;
    }
    proba_state execute(proba_state& in, bool displaytime = false){// initialization and end will take care of repermuting good values
        if (cpumode) return cpusim.execute(displaytime);

        double inittime = -1;
        double measuretime = -1;
        double groupcomputetime = 0;
        int groupnumber = 0;
        double slowswaptime = 0;
        double fastswaptime = 0;
        int fastswapnumber = 0;
        int slowswapnumber = 0;
        
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
                for (const auto& qbit: instr.second){
                    if (qbit < localqbits) continue;
                    if (qbit < nqbits-slowqbitsnumber) {
                        fastswapnumber++;
                        fastswaptime += ms_double.count();
                        continue;
                    }
                    slowswapnumber++;
                    slowswaptime += ms_double.count();

                }
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
            double swaptime = slowswaptime+fastswaptime;
            int swapnumber = slowswapnumber + fastswapnumber;
            std::cout << "Initialization/Measurement : " << inittime << " / " << measuretime << " ms" << std::endl;
            std::cout << "Total computing time : " << swaptime+groupcomputetime << " ms with swap% : " << 100*swaptime/(swaptime+groupcomputetime) << "%" << std::endl;
            std::cout << "Average swap time : " << swaptime/swapnumber << " ms and average group time : " << groupcomputetime/groupnumber << " ms" << std::endl;
            std::cout << "Number swap fast/slow : " << fastswapnumber << "/" << slowqbitsnumber << " Time swap fast/slow : " << fastswaptime << " / " << slowswaptime << " ms" << std::endl;
        }
        return res;
    }
    ~Simulator(){
        if (cpumode) return;

        GPU_CHECK(hipFree(gpu_qbits_state));
        if (number_of_gpu > 1){
            GPU_CHECK(hipFree(swapBuffer1));
            GPU_CHECK(hipFree(swapBuffer2));
        }
        destroyGPUQuantumCircuit(gpuc);
    }
private:
    void initialize(){
        int threadnumber = min(1024llu, (1llu << (nqbits - number_of_gpu_log2)));
        int blocknumber = min((1llu << 20), (1llu << (nqbits - number_of_gpu_log2))/threadnumber);
        if (rank == 0){
            initialize_state<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_state, 0);
        } else {
            initialize_state0<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_state);
        }
        GPU_CHECK(hipDeviceSynchronize());
    }
    void initialize(proba_state& state_input){
        if (state_input.val.size() != nqbits){
            if (rank == 0) std::cout << "wrong input proba_state_size_input, defaulting to no input" << std::endl;
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
        Complex* anglesinter_d;
        Complex offset;
        GPU_CHECK(hipMalloc(&anglesinter_d, sizeof(Complex)*2*(nqbits - number_of_gpu_log2)));
        GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)anglesinter_d, gpustates.data(), sizeof(Complex)*2*(nqbits - number_of_gpu_log2), 0));
        int threadnumber = min(1024llu, (1llu << (nqbits - number_of_gpu_log2)));
        int blocknumber = min((1llu << 20), (1llu << (nqbits - number_of_gpu_log2))/threadnumber);
        offset = Complex(1, 0);
        for (int j = 0; j < number_of_gpu_log2; j++){
            offset = offset * (allstates[((rank >> j)%2)*nqbits + j + nqbits-number_of_gpu_log2]);
        }
        initialize_probastate<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>(nqbits-number_of_gpu_log2, gpu_qbits_state, anglesinter_d, offset);

        GPU_CHECK(hipDeviceSynchronize());
        GPU_CHECK(hipFree(anglesinter_d));   
    }
    proba_state measurement(){
        int threadnumber = min(1024llu, (1llu << (nqbits - number_of_gpu_log2)));
        int blocknumber = min((1llu << 5), (1llu << (nqbits - number_of_gpu_log2))/threadnumber);
        Complex* measureintermediate_d;
        Complex* measureintermediate = (Complex*)malloc(sizeof(Complex)*2*(nqbits-number_of_gpu_log2)*threadnumber*blocknumber);
        std::vector<Complex> measure(2*nqbits, 0);
        Complex temp;
        
        GPU_CHECK(hipMalloc(&measureintermediate_d, sizeof(Complex)*(threadnumber*blocknumber*2*(nqbits-number_of_gpu_log2))));
        measureKernel<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits-number_of_gpu_log2), gpu_qbits_state, measureintermediate_d);
        GPU_CHECK(hipMemcpyDtoHAsync(measureintermediate, (hipDeviceptr_t)measureintermediate_d, sizeof(Complex)*2*(nqbits-number_of_gpu_log2)*threadnumber*blocknumber, 0));

        temp = 0;
        GPU_CHECK(hipDeviceSynchronize());
        GPU_CHECK(hipFree(measureintermediate_d));
        for (int j = 0; j < threadnumber*blocknumber; j++){
            for (int k = 0; k < 2*(nqbits-number_of_gpu_log2); k++){
                measure[k] += measureintermediate[j*2*(nqbits-number_of_gpu_log2) + k];
            }
            temp += measureintermediate[j*2*(nqbits-number_of_gpu_log2) + 0];
            temp += measureintermediate[j*2*(nqbits-number_of_gpu_log2) + 1];
        }
        for (int j = 0; j < number_of_gpu_log2; j++){
            measure[((rank >> j)%2) + 2*(j+(nqbits - number_of_gpu_log2))] += temp;
        }

        free(measureintermediate);
        std::vector<Complex> buffer(2*nqbits);

        //now we need everyone to posses the added copy of measure! for that, we ll use a very nice algorithm!
        int step = 1;
        int lookup = 0;
        MPI_Request sendack;
        while (true){
            lookup = ((rank/step)%2 == 0) ? step : -step;
            if (rank + lookup < 0) break;
            if (rank + lookup >= number_of_gpu) break;
            //we will add our result with rank+lookup peer together
            MPI_Isend((void*)measure.data(), sizeof(Complex)*2*nqbits, MPI_BYTE, rank+lookup, 0, comm, &sendack);
            MPI_Recv((void*)buffer.data(), sizeof(Complex)*2*nqbits, MPI_BYTE, rank+lookup, 0, comm, MPI_STATUS_IGNORE);
            MPI_Wait(&sendack, MPI_STATUS_IGNORE);
            //now we know that measure is free to use and is ours while buffer is measure of the other we can add them up
            for (int qbit = 0; qbit < 2*nqbits; qbit++){
                measure[qbit] += buffer[qbit];
            }

            //and continue until there is no more peer to explore
            step *= 2;
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

        return proba_state(res);
    }
    void swapqbitBufferSwap(int q1, int q2){
        q2 -= nqbits - number_of_gpu_log2;
        size_t data_to_transfer = (1llu << (localqbits - 1));
        size_t chunk_size = std::min((size_t)(1llu << swapBufferSizeLog2), data_to_transfer);

        int peer = rank ^ (1 << q2); //thanks the xor for being so convenient
        int globalindex = (rank >> q2)%2;
        MPI_Request sendack;

        int threadnumber = min(1024llu, (unsigned long long)(chunk_size));
        int blocknumber = min((1llu << 12), (unsigned long long)(chunk_size)/threadnumber);
        int work_per_thread = max(1llu, (unsigned long long)chunk_size/threadnumber/blocknumber);

        for (size_t current = 0; current < data_to_transfer; current += chunk_size){
            //put infos into buffer 1
            swapqbitKernelIndirectAccessEXTRACT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, (1 - globalindex), gpu_qbits_state, swapBuffer1, current, work_per_thread);
            GPU_CHECK(hipDeviceSynchronize());
            //send
            MPI_Isend((void*)swapBuffer1, sizeof(Complex)*chunk_size, MPI_BYTE, peer, 0, comm, &sendack);
            MPI_Recv((void*)swapBuffer2, sizeof(Complex)*chunk_size, MPI_BYTE, peer, 0, comm, MPI_STATUS_IGNORE);
            MPI_Wait(&sendack, MPI_STATUS_IGNORE);
            //import back to memory
            swapqbitKernelIndirectAccessIMPORT<<<dim3(blocknumber), dim3(threadnumber), 0, 0>>>((nqbits - number_of_gpu_log2), q1, (1 - globalindex), gpu_qbits_state, swapBuffer2, current, work_per_thread);
            GPU_CHECK(hipDeviceSynchronize());
        }
    }
    void swapCommand(std::vector<int> pairset){
        for (int i = 0; i < pairset.size()/2; i++){
            int q1 = pairset[2*i];
            int q2 = pairset[2*i+1];
            if (q2 < q1) std::swap(q1, q2);
            swapqbitBufferSwap(q1, q2);
        }
    }
    void executeCommand(int groupind){
        std::set<int> newqbits = groups[groupind].second;
        //we will add some qbits to make use of a block. Ideally, we should have 9
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

        int* groupqbitsgpu;
        GPU_CHECK(hipMalloc(&groupqbitsgpu, sizeof(int)*qbits.size()));
        GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)groupqbitsgpu, qbits.data(), sizeof(int)*qbits.size(), 0));
    
        hipDeviceProp_t devattr;
        int device;
        GPU_CHECK(hipGetDevice(&device));
        GPU_CHECK(hipGetDeviceProperties(&devattr, device));
        size_t totalshared_block = devattr.sharedMemPerBlock/4;
        int threadnumber = std::min(1024llu, (1llu << (qbits.size())));
        int blocknumber = std::min((1llu << 20), (1llu << ((nqbits - number_of_gpu_log2) - qbits.size())));
        if ((1llu << qbits.size()) > totalshared_block){
            std::cout << "too much qbits in one group for this gpu's shared memory... I cancel this group's computation" << std::endl;
        }
        executeGroupKernelSharedState<<<dim3(blocknumber), dim3(threadnumber), totalshared_block, 0>>>((nqbits - number_of_gpu_log2), gpu_qbits_state, qbits.size(), groupqbitsgpu, gpuc.gates+i, j-i, totalshared_block - sizeof(Complex)*(1llu << qbits.size()));
        hipError_t kernelret = hipGetLastError();
        if (kernelret != hipSuccess){
            std::cout << "Error while trying to compute a group, the  kernel refused to launch for this reason : " << std::endl << hipGetErrorString(kernelret) << std::endl << "It could be due to a lack of shared cache, please reduce the size of each group" << std::endl;
        }

        GPU_CHECK(hipDeviceSynchronize());
        GPU_CHECK(hipFree(groupqbitsgpu));
    }
};

}

#endif