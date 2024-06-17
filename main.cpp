#include "preprocessor.hpp"
#include "QuantumCircuit.hpp"
#include "simulator.hpp"
#include "BenchCircuits.hpp"
#include "DeviceInfo.hpp"

int main(){
    printGpuInfo<Complex<double>>();

    /*
    auto circuit = QulacsBench(6);
    circuit.gateScheduling();
    circuit.gateFusion(5, 0.00001);
    circuit.gateGrouping(2);

    circuit.optimal_slow_fast_allocation(2, 1, 32, 1300);
    //circuit.allocate(3);
    //circuit.dual_phase_allocation(2, 1);

    circuit.print();*/

    ///*
    int nqbits = 33;
    //int slowqbits = 8;
    int fastqbits = 3;

    for (int slowqbits = 0; slowqbits < 20; slowqbits++){
        auto circuit = QFT<double>(nqbits);
        //slowqbits = nqbits/3;
        //fastqbits = 3;
        circuit.gateScheduling();
        circuit.gateFusion(5, 0.00001);
        circuit.gateGrouping(2);

        circuit.dual_phase_allocation(fastqbits, slowqbits); //2 slow swaps with 18 total swaps
        circuit.allocate(slowqbits + fastqbits); //7 slow swap with 17 total swaps
        circuit.slow_fast_allocation(fastqbits, slowqbits, 32, 1350);

        //circuit.print();

        int slowswap = 0;
        int fastswap = 0;
        for (const auto& instr: circuit.instructions){
            if (instr.first == 0){
                for (int i = 0; i+1 < instr.second.size(); i+= 2){
                    int first = instr.second[i];
                    int second = instr.second[i+1];
                    if (first >= nqbits-slowqbits) {slowswap++;continue;}
                    if (second >= nqbits-slowqbits) {slowswap++;continue;}
                    fastswap++;
                }
            }
        }

        cout << "fast swap : " << fastswap << "; slow swap : " << slowswap << "; total swap : " << fastswap+slowswap << endl; 
    }//*/
    cout << "done" << endl;

    return 0;
}
