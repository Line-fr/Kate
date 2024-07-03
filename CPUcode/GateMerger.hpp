#ifndef CPUGATEMERGERDONE
#define CPUGATEMERGERDONE

#include "GateComputing.hpp"

namespace Kate{

Gate CPUmergeGate(std::vector<Gate> to_merge){
    int maxseen = 0;
    std::set<int> total_covered;
    for (const auto& gate: to_merge){
        for (const auto& qbit: gate.qbits){
            total_covered.insert(qbit);
            if (qbit > maxseen) maxseen = qbit;
        }
    }

    std::vector<int> coveredqbits_ordered(total_covered.size());
    int i = 0;
    for (const auto& el: total_covered){
        coveredqbits_ordered[i] = el;
        i++;
    }

    std::vector<int> bit_to_groupbitnumber(maxseen+1);
    for (int i = 0; i < total_covered.size(); i++){
        bit_to_groupbitnumber[coveredqbits_ordered[i]] = i;
    }

    Matrix<Complex> newdense(1llu << total_covered.size());

    //init is diagonal
    for (int i = 0; i < (1llu << total_covered.size()); i++){
        for (int j = 0; j < (1llu << total_covered.size()); j++){
            newdense(i, j, (int)(i==j));
        }
    }

    int maxgatesize = 0;
    for (const auto& gate: to_merge){
        if (gate.qbits.size() > maxgatesize) maxgatesize = gate.qbits.size();
    }

    for (int i = 0; i < (1llu << total_covered.size());i++){
        //i is matrix line representing column in the computation here
        for (const auto& gate:to_merge){
            std::vector<int> orderedgateqbits = gate.qbits;
            std::sort(orderedgateqbits.begin(), orderedgateqbits.end());
            computeGate(gate, total_covered.size(), newdense.data+i*newdense.n, bit_to_groupbitnumber.data(), orderedgateqbits, std::vector<Complex>((1llu << maxgatesize)));
        }
    }

    newdense.transpose(); //we wrote it by line, but we work with column
    return Gate(newdense, coveredqbits_ordered);
}

}

#endif