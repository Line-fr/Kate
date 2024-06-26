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
    

    //Circuit circuit = QulacsBench(qbitnumber);
    //circuit.compileOPT(5, 0.00001, 9, gpulog2);
    //Simulator(circuit, (1 << gpulog2)).execute(true).print();

    for (int i = 5; i < 28; i++){
        Circuit circuit = QulacsBench(i);
        circuit.compileOPT(5, 0.00001, 9, gpulog2);
        std::cout << "GPU" << std::endl;
        Simulator(circuit, (1 << gpulog2)).execute(true);
        cout << "CPU" << endl;
        CPUSimulator(circuit, 0).execute(true);
    }

    cout << "done" << endl;

    return 0;
}
