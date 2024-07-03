#ifndef PREPROCESSORDONE
#define PREPROCESSORDONE

#include<mpi.h>

#define INITIALIZER MPI_Init(&argc, &argv);
#define FINALIZER MPI_Finalize();

#include "backendagnosticcode/agnosticpreprocessor.hpp"
#include "backendagnosticcode/graphUtil.hpp"
#include "backendagnosticcode/basic_host_types.hpp"
#include "backendagnosticcode/QuantumCircuit.hpp"

#include "CPUcode/CPUpreprocessor.hpp"
#include "CPUcode/GateComputing.hpp"
#include "CPUcode/GateMerger.hpp"

#include "MPIcode/MPIpreprocessor.hpp"
#include "MPIcode/DeviceInfo.hpp"

#include "MPIcode/simulator.hpp"

#include "backendagnosticcode/Circuit.hpp"

#include<vector>
#include<set>
#include<math.h>
#include<limits.h>
#include<algorithm>
#include<list>
#include<chrono>
#include<stdlib.h>
#include<iostream>
#include<queue>
#include<iomanip>
#include<cstring>
#include<thread>

#define Hadamard 2
#define CNOT 3
#define CRk 4
#define TOFFOLI 5
#define RX 6
#define RZ 7

#define THREADNUMBER 64
#define GPUTHREADSNUM 1024

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#endif