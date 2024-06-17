#include "preprocessor.hpp"
#include "Circuit.hpp"
#include "simulator.hpp"
#include "BenchCircuits.hpp"
#include "DeviceInfo.hpp"

int main(){
    printGpuInfo();

    auto circuit = QulacsBench(6);
    circuit.compileOPT();
    Simulator(circuit, 1).execute(true).print();

    cout << "done" << endl;

    return 0;
}
