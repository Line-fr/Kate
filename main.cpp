#include "Kate.hpp"
#include "BenchCircuits.hpp"

using namespace Kate;
using namespace std;

int main(int argc, char** argv){
    if (argc != 3){
        cout << "usage : " << "./a.out [number of qubits] [number of gpu log2]" << endl;
        return 42;
    }
    int gpulog2 = atoi(argv[2]);
    int qbitnumber = atoi(argv[1]);

    printGpuInfo();
    
    Circuit circuit = QulacsBench(qbitnumber);
    //Circuit circuit(qbitnumber);
    //vector<int> qbits = {};
    //for (int i = 0; i < qbitnumber; i++){
    //    qbits = {i};
    //    circuit.appendGate(Gate(Hadamard, qbits));
    //}
    circuit.compileOPT(5, 0.00001, 9, gpulog2);
    //circuit.print();
    Simulator(circuit, (1 << gpulog2)).execute(true).print();

    cout << "done" << endl;

    return 0;
}
