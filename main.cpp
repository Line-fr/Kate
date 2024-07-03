#include "Kate.hpp"
//#include "KateMPI.hpp"
#include "BenchCircuits.hpp"

using namespace Kate;
using namespace std;

int main(int argc, char** argv){
    INITIALIZER
    if (argc != 3){
        cout << "usage : " << "./a.out [number of qubits] [number of gpu log2]" << endl;
        return 42;
    }
    int gpulog2 = atoi(argv[2]);
    int qbitnumber = atoi(argv[1]);

    printGpuInfo();
    
    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Circuit circuit = QulacsBench(qbitnumber);
    //Circuit circuit(qbitnumber);
    //vector<int> qbits;
    //for (int i = 0; i < qbitnumber; i++){
    //    qbits = {i};
    //    circuit.appendGate(Gate(Hadamard, qbits));
    //}
    //circuit.compileOPT(5, 1, 9, gpulog2);
    circuit.gateScheduling();
    //circuit.gateFusion(5 , 1);
    circuit.gateGrouping(12);
    circuit.allocate(gpulog2);
    
    //if (rank == 0) circuit.print();
    auto res = Simulator(circuit, MPI_COMM_WORLD).execute(true);
    //if (rank == 0) res.print();

    //circuit.exportqcx("testcircuit.qcx");

    //if (rank == 0) cout << "done" << endl;

    FINALIZER

    return 0;
}
