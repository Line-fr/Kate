#ifndef QUANTUMCIRCUITDONE
#define QUANTUMCIRCUITDONE

#include "graphUtil.hpp"
#include "preprocessor.hpp"
#include "basic_host_types.hpp"
#include "GPUMatrix.hpp"
#include "GPUGate.hpp"

class QuantumCircuit{
public:
    vector<Gate> gate_set_ordered;
    int nqbits = 0;
    vector<pair<int, set<int>>> groups; //second is current qbit set, first is when to go to next group
    vector<int> initial_permutation;
    vector<int> final_inverse_permutation;
    vector<pair<int, vector<int>>> instructions; //contains either 0 for swap and some qbits (they go by pair) or 1 for compute (just compute next group available)
    QuantumCircuit(int nqbits){
        this->nqbits = nqbits;
    }
    QuantumCircuit(const vector<Gate>& gate_set, int nqbits){
        this->nqbits = nqbits;
        gate_set_ordered = gate_set;
    }
    void appendGate(const Gate& gate){
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
};

#endif