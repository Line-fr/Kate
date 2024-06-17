#ifndef QUANTUMCIRCUITDONE
#define QUANTUMCIRCUITDONE

#include "graphUtil.hpp"
#include "preprocessor.hpp"
#include "basic_host_types.hpp"
#include "GPUMatrix.hpp"
#include "GPUGate.hpp"
#include "GateMerger.hpp"

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
            Gate<T> ngate = mergeGate<T>(to_merge);

            newcircuit.push_back(ngate);

            f = el;
        }
        gate_set_ordered = newcircuit;
    }
    void gateGrouping(int qbitgroupsize = 8){ // OPTIMISATION STEP 3 (optimal given a schedule)
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

#endif