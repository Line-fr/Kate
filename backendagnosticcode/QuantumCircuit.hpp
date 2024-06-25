#ifndef QUANTUMCIRCUITDONE
#define QUANTUMCIRCUITDONE

#include "graphUtil.hpp"
#include "agnosticpreprocessor.hpp"
#include "basic_host_types.hpp"

namespace Kate {

class QuantumCircuit{
public:
    std::vector<Gate> gate_set_ordered;
    int nqbits = 0;
    std::vector<std::pair<int, std::set<int>>> groups; //second is current qbit set, first is when to go to next group
    std::vector<int> initial_permutation;
    std::vector<int> final_inverse_permutation;
    std::vector<std::pair<int, std::vector<int>>> instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
    QuantumCircuit(int nqbits){
        this->nqbits = nqbits;
    }
    QuantumCircuit(const std::vector<Gate>& gate_set, int nqbits){
        this->nqbits = nqbits;
        gate_set_ordered = gate_set;
    }
    void appendGate(const Gate& gate){
        //checking qbits
        for (const auto& el: gate.qbits){
            if (el >= nqbits){
                std::cout << "the gate you are trying to add contains a qbit not in the circuit: " << el << "/" << nqbits << "!" << std::endl;
                return;
            }
        }
        if (groups.size() != 0) {
            std::cout << "you should not be adding gate after gateGrouping or allocate, this will not work" << std::endl;
        }
        gate_set_ordered.push_back(gate);
    }
    void print(bool permutationtable = 0){
        int i = 0;
        int group = 0;
        if (instructions.size() == 0){
            if (groups.size() != 0){
                std::cout << "Group 0 with qbits ";
                for (const auto& el2: groups[group].second){
                    std::cout << el2 << " ";
                }
                std::cout << " : " << std::endl;
            }
            for (auto& el: gate_set_ordered){
                if (groups.size() != 0){
                    if (group >= groups.size()){
                        std::cout << "ERROR while printing the circuit, group vector is bad" << std::endl;
                        return;
                    }
                    if (groups[group].first == i) {
                        group++;
                        std::cout << "Group " << group << " with qbits ";
                        for (const auto& el2: groups[group].second){
                            std::cout << el2 << " ";
                        }
                        std::cout << " : " << std::endl;
                    }
                    std::cout << "   ";
                    el.print();
                } else {
                    std::cout << "gate " << i << " is ";
                    el.print();
                }
                i++;
            }
        } else {
            std::vector<int> permutation = initial_permutation;
            std::vector<int> inversepermutation(nqbits);
            for (int m = 0; m < nqbits; m++){
                if (permutation[m] >= nqbits){
                    std::cout << "Error while printing: initial permutation work on more qbits than the number in the circuit!" << std::endl;
                    return;
                }
                inversepermutation[permutation[m]] = m;
            }

            //print when we did allocate
            std::cout << "-------Quantum-Program-------" << std::endl;
            for (const auto& instruction: instructions){
                if (instruction.first == 0){
                    std::cout << "SWAP ";
                    for (int m = 0; m < instruction.second.size()/2; m++){
                        std::cout << instruction.second[2*m] << " and " << instruction.second[2*m+1] << ", ";

                        std::swap(inversepermutation[instruction.second[2*m]], inversepermutation[instruction.second[2*m+1]]);
                        std::swap(permutation[inversepermutation[instruction.second[2*m]]], permutation[inversepermutation[instruction.second[2*m+1]]]);
                    }
                } else{
                    if (permutationtable){
                        std::cout << "Permutation table (subjective to real) : " << std::endl;
                        for (int m = 0; m < nqbits; m++){
                            std::cout << m << " ";
                            if (m < 10) std::cout << " ";
                        }
                        std::cout << std::endl;
                        for (int m = 0; m < nqbits; m++){
                            std::cout << inversepermutation[m] << " ";
                            if (inversepermutation[m] < 10) std::cout << " ";
                        }
                        std::cout << std::endl;
                        std::cout << std::endl;
                    }

                    std::cout << "EXEC Group " << group << " with qbits ";
                    for (const auto& el2: groups[group].second){
                        std::cout << el2 << " ";
                    }
                    std::cout << " : " << std::endl;
                    for (int j = i; j < groups[group].first; j++){
                        std::cout << "   ";
                        gate_set_ordered[j].print();
                    }
                    i = groups[group].first;
                    group++;
                }
                std::cout << std::endl << std::endl;
            }
            std::cout << "-----------------------------" << std::endl;
        }
    }
    void compileDefault(int numberofgpulog2 = 0, int maxlocalqbitnumber = 300){ //for every optimization that hasnt been done but was necessary, it will use naive things to replace them
        //only support homogeneous gpus or the slow one will slow the big one
        if (instructions.size() != 0) return; // case where everything has already been done (possible that allocate was optimized but not grouping)
        
        if (maxlocalqbitnumber + numberofgpulog2 < nqbits){
            std::cout << "Error: Can't allocate - Too much qbits in the circuit to handle with " << maxlocalqbitnumber << " localqbits and " << (1llu << numberofgpulog2) << " gpus" << std::endl;
            return;
        }
        
        maxlocalqbitnumber = nqbits - numberofgpulog2; //this line is to modify the behaviour from "use fewest amount of gpu to as much as permitted by options"
        //comment this line to come back to old behaviour
        if (groups.size() == 0){
            for (int i = 0; i < gate_set_ordered.size(); i++){
                groups.push_back(std::make_pair(i+1, std::set<int>(gate_set_ordered[i].qbits.begin(), gate_set_ordered[i].qbits.end())));
            }
        }
        //now we need the naive allocation (allocate necessary qbit to the most left qbit)
        std::vector<int> permutation(nqbits, 0); //super important
        std::vector<int> inversepermutation(nqbits, 0);
        for (int i = 0; i < nqbits; i++){
            permutation[i] = i;
            inversepermutation[i] = i;
        }

        initial_permutation = permutation;

        int k = 0;
        std::set<int> alreadytaken;
        std::vector<std::pair<int, int>> pairs;
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
                    pairs.push_back(std::make_pair(permutation[qbit], permutation[newlocal]));
                    //now let's refresh permutations
                    std::swap(inversepermutation[permutation[qbit]], inversepermutation[permutation[newlocal]]);
                    std::swap(permutation[qbit], permutation[newlocal]);
                }
            }
            if (pairs.size() != 0){
                //swap operation needed!
                std::vector<int> pairsset;
                for (const auto& pair: pairs){
                    pairsset.push_back(pair.first);
                    pairsset.push_back(pair.second);
                }
                instructions.push_back(std::make_pair(0, pairsset));
            }
            instructions.push_back(std::make_pair(1, std::vector<int>()));
            //we need to modify gates subjective qbits and of the group
            std::set<int> temp;
            for (const auto& el: groups[group].second){
                temp.insert(permutation[el]);
            }
            groups[group].second = temp;
            std::vector<int> temp2;
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
};

}

#endif