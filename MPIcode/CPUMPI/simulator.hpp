#ifndef CPUSIMULATORDONE
#define CPUSIMULATORDONE

namespace Kate {

class CPUSimulator{
public:
    //copy from quantum circuit but with the gpu version
    std::vector<std::pair<int, std::set<int>>> groups; //second is current qbit set, first is when to go to next group
    std::vector<int> initial_permutation;
    std::vector<int> final_inverse_permutation;
    std::vector<std::pair<int, std::vector<int>>> instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
    std::vector<Gate> gate_set_ordered;
    int nqbits = 0;

    int number_of_gpu = 0;
    int number_of_gpu_log2 = 0;
    int localqbits = 0;
    Complex* gpu_qbits_state = NULL;

    Complex* swapBuffer1 = NULL;
    Complex* swapBuffer2 = NULL;
    int swapBufferSizeLog2 = 24;

    int rank = 0;
    MPI_Comm comm = MPI_COMM_WORLD;

    CPUSimulator(){
    }

    CPUSimulator(QuantumCircuit mycircuit, MPI_Comm comm, int swapBufferSizeLog2 = 24){
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &number_of_gpu);
        this->comm = comm;

        if (mycircuit.instructions.size() == 0){
            if (rank == 0) std::cout << "warning: the simulator has been input a circuit that is not compiled. I will compile it naively now" << std::endl;
            mycircuit.compileDefault((int)log2(number_of_gpu), mycircuit.nqbits - (int)log2(number_of_gpu));
        }
        groups = mycircuit.groups;
        initial_permutation = mycircuit.initial_permutation;
        final_inverse_permutation = mycircuit.final_inverse_permutation;
        instructions = mycircuit.instructions;
        gate_set_ordered = mycircuit.gate_set_ordered;
        nqbits = mycircuit.nqbits;

        number_of_gpu_log2 = (int)log2(number_of_gpu);
        number_of_gpu = (1llu << number_of_gpu_log2); //only use power of 2;
        this->swapBufferSizeLog2 = swapBufferSizeLog2;
        localqbits = nqbits - number_of_gpu_log2;
        
        gpu_qbits_state = (Complex*)malloc(sizeof(Complex)*(1llu << (localqbits)));
        if (number_of_gpu > 1){
            swapBuffer1 = (Complex*)malloc(sizeof(Complex)*(1llu << swapBufferSizeLog2));
            swapBuffer2 = (Complex*)malloc(sizeof(Complex)*(1llu << swapBufferSizeLog2));
        }
    }
    proba_state execute(bool displaytime = false){// initialization and end will take care of repermuting good values
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

        if (displaytime && rank == 0){
            std::cout << "Initialization/Measurement : " << inittime << " / " << measuretime << " ms" << std::endl;
            std::cout << "Total computing time : " << swaptime+groupcomputetime << " ms with swap% : " << 100*swaptime/(swaptime+groupcomputetime) << "%" << std::endl;
            std::cout << "Average swap time : " << swaptime/swapnumber << " ms and average group time : " << groupcomputetime/groupnumber << " ms" << std::endl;
        }
        return res;
    }
    proba_state execute(proba_state& in, bool displaytime = false){// initialization and end will take care of repermuting good values
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

        if (displaytime && rank == 0){
            std::cout << "Initialization/Measurement : " << inittime << " / " << measuretime << " ms" << std::endl;
            std::cout << "Total computing time : " << swaptime+groupcomputetime << " ms with swap% : " << 100*swaptime/(swaptime+groupcomputetime) << "%" << std::endl;
            std::cout << "Average swap time : " << swaptime/swapnumber << " ms and average group time : " << groupcomputetime/groupnumber << " ms" << std::endl;
        }
        return res;
    }
    ~CPUSimulator(){
        if (gpu_qbits_state == NULL) return;
        free(gpu_qbits_state);
        if (number_of_gpu > 1){
            free(swapBuffer1);
            free(swapBuffer2);
        }
    }
private:
    void initialize(){
        for (int i = 0; i < (1llu << localqbits); i++){
            gpu_qbits_state[i] = 0;
        }
        if (rank == 0) gpu_qbits_state[0] = 1;
    }
    void initialize(proba_state& state_input){
        if (state_input.val.size() != nqbits){
            std::cout << "wrong input proba_state_size_input, defaulting to no input" << std::endl;
            initialize();
            return;
        }
        std::vector<Complex> allstates(2*nqbits);
        for (int i = 0; i < nqbits; i++){
            Complex val0, val1;
            val0 = Complex(cos((state_input.val[i].first)*PI/2), 0);
            val1 = Complex(cos((state_input.val[i].second)), sin((state_input.val[i].second)))*sin((state_input.val[i].first)*PI/2);
            allstates[initial_permutation[i]] = val0;
            allstates[initial_permutation[i] + nqbits] = val1;
        }

        Complex offset;
        offset = Complex(1, 0);
        for (int j = 0; j < number_of_gpu_log2; j++){
            offset = offset * (allstates[((rank >> j)%2)*nqbits + j + nqbits-number_of_gpu_log2]);
        }

        Complex temp;
        for (int i = 0; i < (1llu << localqbits); i++){
            temp = offset;
            for (int j = 0; j < localqbits; j++){
                temp *= allstates[((i >> j)%2)*nqbits + j];
            }
            gpu_qbits_state[i] = temp;
        }
    }
    proba_state measurement(){
        std::vector<Complex> measure(2*nqbits, 0);
        std::vector<Complex> buffer(2*nqbits);

        for (int i = 0; i < (1llu << localqbits); i++){
            for (int qbit = 0; qbit < localqbits; qbit++){
                measure[qbit*2 + ((i >> qbit)%2)] += gpu_qbits_state[i];
            }
        }
        for (int qbit = localqbits; qbit < nqbits; qbit++){
            measure[qbit*2 + ((rank >> (qbit-localqbits))%2)] += measure[0] + measure[1];
        }

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

        size_t mask = (1llu << q1) - 1;
        size_t mask2 = (1llu << (localqbits - 1)) - 1 - mask;

        for (size_t current = 0; current < data_to_transfer; current += chunk_size){
            //put infos into buffer 1
            for (int i = current; i < current+chunk_size; i++){
                size_t value = (i&mask) + ((i&mask2) << 1);
                swapBuffer1[i-current] = gpu_qbits_state[value + ((1 - globalindex) << q1)];
            }
            //send
            MPI_Isend((void*)swapBuffer1, sizeof(Complex)*chunk_size, MPI_BYTE, peer, 0, comm, &sendack);
            MPI_Recv((void*)swapBuffer2, sizeof(Complex)*chunk_size, MPI_BYTE, peer, 0, comm, MPI_STATUS_IGNORE);
            MPI_Wait(&sendack, MPI_STATUS_IGNORE);
            //import back to memory
            for (int i = current; i < current+chunk_size; i++){
                size_t value = (i&mask) + ((i&mask2) << 1);
                gpu_qbits_state[value + ((1 - globalindex) << q1)] = swapBuffer2[i-current];
            }
        }
    }
    void globalswapqbitBufferSwap(int q1, int q2){
        q2 -= nqbits - number_of_gpu_log2;
        q1 -= nqbits - number_of_gpu_log2;
        size_t data_to_transfer = (1llu << (localqbits));
        size_t chunk_size = std::min((size_t)(1llu << swapBufferSizeLog2), data_to_transfer);

        int peer = rank ^ (1 << q2) ^ (1 << q1); //thanks the xor for being so convenient
        int isconcerned = ((rank >> q1) + (rank >> q2))%2;
        if (isconcerned == 0) return;
        MPI_Request sendack;

        for (size_t current = 0; current < data_to_transfer; current += chunk_size){
            //put infos into buffer 1
            for (int i = current; i < current+chunk_size; i++){
                swapBuffer1[i-current] = gpu_qbits_state[i];
            }
            //send
            MPI_Isend((void*)swapBuffer1, sizeof(Complex)*chunk_size, MPI_BYTE, peer, 0, comm, &sendack);
            MPI_Recv((void*)swapBuffer2, sizeof(Complex)*chunk_size, MPI_BYTE, peer, 0, comm, MPI_STATUS_IGNORE);
            MPI_Wait(&sendack, MPI_STATUS_IGNORE);
            //import back to memory
            for (int i = current; i < current+chunk_size; i++){
                gpu_qbits_state[i] = swapBuffer2[i-current];
            }
        }
    }
    void swapCommand(std::vector<int> pairset){
        for (int i = 0; i < pairset.size()/2; i++){
            int q1 = pairset[2*i];
            int q2 = pairset[2*i+1];
            if (q2 < q1) std::swap(q1, q2);
            if (q1 >= localqbits){
                //slow fast swap
                globalswapqbitBufferSwap(q1, q2);
            } else {
                swapqbitBufferSwap(q1, q2);
            }
        }
    }
    void executeCommand(int groupind){
        std::set<int> newqbits = groups[groupind].second;
        //we will add some qbits to make use of a block. Ideally, we should have at least 10
        for (int l = 0; l < (localqbits); l++){
            if (newqbits.size() >= 14 || newqbits.size() == (localqbits)) break;
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

        int groupnqbits = newqbits.size();
        auto groupqbits = qbits;

        std::vector<int> bit_to_groupbitnumber(localqbits);
        for (int i = 0; i < groupnqbits; i++){
            bit_to_groupbitnumber[groupqbits[i]] = i;
        }
        
        size_t mask_group[64];
        size_t cumulative = 0;
        for (int i = 0; i < groupnqbits; i++){
            mask_group[i] = (1llu << (groupqbits[i] - i)) - 1 - cumulative;
            cumulative += mask_group[i];
        }
        mask_group[groupnqbits] = (1llu << (localqbits - groupnqbits)) - 1 - cumulative;

        size_t groupnumber = (1llu << (localqbits - groupnqbits));

        std::vector<std::vector<int>> orderedgateqbits;
        for (const auto& gate: gate_set_ordered){
            orderedgateqbits.push_back(gate.qbits);
            std::sort(orderedgateqbits[orderedgateqbits.size()-1].begin(), orderedgateqbits[orderedgateqbits.size()-1].end());
        }

        int maxgatesize = 0;
        for (const auto& gate: gate_set_ordered){
            if (gate.qbits.size() > maxgatesize) maxgatesize = gate.qbits.size();
        }

        auto threadwork = [&](int g, int h){
            std::vector<Complex> cache((1llu << maxgatesize));
            std::vector<Complex> qbitsstateshared((1llu << groupnqbits));
            for (size_t groupel = g; groupel < h; groupel++){
            
                size_t groupbaseind = 0;
                for (int m = 0; m <= groupnqbits; m++){
                    groupbaseind += ((groupel&mask_group[m]) << m);
                } // XXXX-0-XXXXX-0-XX... 0 for all qbit group of the group
            
                for (int line = 0; line < (1llu << groupnqbits); line++){
                    size_t finalbaseind = groupbaseind;
                    for (int i = 0; i < groupnqbits; i++){
                        finalbaseind += ((line >> i)%2) << groupqbits[i];
                    }
                
                    qbitsstateshared[line] = gpu_qbits_state[finalbaseind];
                    //printf("value at line: %i is %f with finalbaseind : %i\n", line, qbitsstateshared[line].a, (int)finalbaseind);
                }

                for (int gateid = i; gateid < j; gateid++){
                    computeGate(gate_set_ordered[gateid], groupnqbits, qbitsstateshared.data(), bit_to_groupbitnumber.data(), orderedgateqbits[gateid], cache);
                }

                for (int line = 0; line < (1llu << groupnqbits); line++){
                    size_t finalbaseind = groupbaseind;
                    for (int m = 0; m < groupnqbits; m++){
                        finalbaseind += ((line >> m)%2) << groupqbits[m];
                    }
                    gpu_qbits_state[finalbaseind] = qbitsstateshared[line];
                }
            }
        };

        
        std::vector<std::thread> threads;
        int threadnumber = std::min(groupnumber, (size_t)std::thread::hardware_concurrency());

        //threadnumber = 1;

        size_t work_per_thread = groupnumber/threadnumber;
        for (int th = 0; th < threadnumber; th++){
            threads.emplace_back(threadwork, (int)(th*work_per_thread), (int)std::min((th+1)*work_per_thread, groupnumber));
        }
        for (int th = 0; th < threadnumber; th++){
            threads[th].join();
        }
    }
};

}

#endif