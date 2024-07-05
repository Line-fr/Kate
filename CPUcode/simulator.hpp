#ifndef CPUSIMULATORDONE
#define CPUSIMULATORDONE

#include "GateComputing.hpp"

namespace Kate{
    
class CPUSimulator{
public:
    //copy from quantum circuit but with the gpu version
    std::vector<std::pair<int, std::set<int>>> groups; //second is current qbit set, first is when to go to next group
    std::vector<int> initial_permutation;
    std::vector<int> final_inverse_permutation;
    std::vector<std::pair<int, std::vector<int>>> instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
    std::vector<Gate> gate_set_ordered;
    int nqbits = 0;

    Complex* qbitstate = NULL;

    CPUSimulator(){
    }
    CPUSimulator(QuantumCircuit mycircuit, int number_of_gpu = 0, int swapBufferSizeLog2 = 24){
        if (number_of_gpu != 0){
            std::cout << "Warning: Running simulation on CPU" << std::endl;
        }
        number_of_gpu = 1;
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

        qbitstate = (Complex*)malloc(sizeof(Complex)*(1llu << nqbits));
        if (qbitstate == NULL){
            std::cout << "Error: Failed allocation of vector, probably there is not enough memory" << std::endl;
        }
    }
    void operator=(const CPUSimulator& other){        
        groups = other.groups; //second is current qbit set, first is when to go to next group
        initial_permutation = other.initial_permutation;
        final_inverse_permutation = other.final_inverse_permutation;
        instructions = other.instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
        gate_set_ordered = other.gate_set_ordered;
        nqbits = other.nqbits;

        if (qbitstate != NULL) free(qbitstate);
        qbitstate = (Complex*)malloc(sizeof(Complex)*(1llu << nqbits));
        if (qbitstate == NULL){
            std::cout << "Error: Failed allocation of vector, probably there is not enough memory" << std::endl;
        }
    }
    void operator=(const CPUSimulator&& other){
        groups = other.groups; //second is current qbit set, first is when to go to next group
        initial_permutation = other.initial_permutation;
        final_inverse_permutation = other.final_inverse_permutation;
        instructions = other.instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
        gate_set_ordered = other.gate_set_ordered;
        nqbits = other.nqbits;

        if (qbitstate != NULL) free(qbitstate);
        qbitstate = (Complex*)malloc(sizeof(Complex)*(1llu << nqbits));
        if (qbitstate == NULL){
            std::cout << "Error: Failed allocation of vector, probably there is not enough memory" << std::endl;
        }
    }
    proba_state execute(bool displaytime = false){// initialization and end will take care of repermuting good values
        auto t1 = high_resolution_clock::now();
        initialize();
        auto t2 = high_resolution_clock::now();

        int groupid = 0;
        for (const auto& instr: instructions){
            if (instr.first == 0){
                std::cout << "Warning: no swap supported on CPU version" << std::endl;
            } else if (instr.first == 1){
                executeCommand(groupid);
                groupid++;
            }
        }
        auto t3 = high_resolution_clock::now();
        auto res = measurement();
        auto t4 = high_resolution_clock::now();

        duration<double, std::milli> ms_double_init = t2 - t1;
        duration<double, std::milli> ms_double_compute = t3 - t2;
        duration<double, std::milli> ms_double_end = t4 - t3;

        if (displaytime){
            std::cout << "Initialization time : " << ms_double_init.count() << " ms" << std::endl;
            std::cout << "Computation time : " << ms_double_compute.count() << " ms" << std::endl;
            std::cout << "measurement time : " << ms_double_end.count() << " ms" << std::endl;
        }
        return res;
    }
    proba_state execute(proba_state& in, bool displaytime = false){// initialization and end will take care of repermuting good values
        auto t1 = high_resolution_clock::now();
        initialize(in);
        auto t2 = high_resolution_clock::now();

        int groupid = 0;
        for (const auto& instr: instructions){
            if (instr.first == 0){
                std::cout << "Warning: no swap supported on CPU version" << std::endl;
            } else if (instr.first == 1){
                executeCommand(groupid);
                groupid++;
            }
        }
        auto t3 = high_resolution_clock::now();
        auto res = measurement();
        auto t4 = high_resolution_clock::now();

        duration<double, std::milli> ms_double_init = t2 - t1;
        duration<double, std::milli> ms_double_compute = t3 - t2;
        duration<double, std::milli> ms_double_end = t4 - t3;

        if (displaytime){
            std::cout << "Initialization time : " << ms_double_init.count() << " ms" << std::endl;
            std::cout << "Computation time : " << ms_double_compute.count() << " ms" << std::endl;
            std::cout << "measurement time : " << ms_double_end.count() << " ms" << std::endl;
        }
        return res;
    }
    ~CPUSimulator(){
        if (qbitstate != NULL) free(qbitstate);
    }
private:
    void initialize(){
        qbitstate[0] = 1;
        for (size_t i = 1; i < (1llu << nqbits); i++){
            qbitstate[i] = 0;
        }
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

        Complex temp;
        for (size_t i = 0; i < (1llu << nqbits); i++){
            temp = 1.;
            for (int j = 0; j < nqbits; j++){
                temp *= allstates[((i >> j)%2)*nqbits + j];
            }
            qbitstate[i] = temp;
        }
    }
    proba_state measurement(){
        Complex* measure = (Complex*)malloc(sizeof(Complex)*2*nqbits);
        for (int i = 0; i < nqbits*2; i++){
            measure[i] = 0;
        }

        for (int i = 0; i < (1llu << nqbits); i++){
            for (int qbit = 0; qbit < nqbits; qbit++){
                measure[qbit*2 + ((i >> qbit)%2)] += qbitstate[i];
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

        free(measure);

        return proba_state(res);
    }
    void executeCommand(int groupind){
        std::set<int> newqbits = groups[groupind].second;
        //we will add some qbits to make use of a block. Ideally, we should have at least 10
        for (int l = 0; l < (nqbits); l++){
            if (newqbits.size() >= 14 || newqbits.size() == (nqbits)) break;
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

        std::vector<int> bit_to_groupbitnumber(nqbits);
        for (int i = 0; i < groupnqbits; i++){
            bit_to_groupbitnumber[groupqbits[i]] = i;
        }
        
        size_t mask_group[64];
        size_t cumulative = 0;
        for (int i = 0; i < groupnqbits; i++){
            mask_group[i] = (1llu << (groupqbits[i] - i)) - 1 - cumulative;
            cumulative += mask_group[i];
        }
        mask_group[groupnqbits] = (1llu << (nqbits - groupnqbits)) - 1 - cumulative;

        size_t groupnumber = (1llu << (nqbits - groupnqbits));

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
                
                    qbitsstateshared[line] = qbitstate[finalbaseind];
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
                    qbitstate[finalbaseind] = qbitsstateshared[line];
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