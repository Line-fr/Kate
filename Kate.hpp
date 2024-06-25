#ifndef PREPROCESSORDONE
#define PREPROCESSORDONE

#include "backendagnosticcode/agnosticpreprocessor.hpp"
#include "backendagnosticcode/graphUtil.hpp"
#include "backendagnosticcode/basic_host_types.hpp"
#include "backendagnosticcode/QuantumCircuit.hpp"

#if ((!(defined __HIPCC__)) && (!defined __NVCC__) && (!defined _OPENMP))
    #include "CPUcode/DeviceInfo.hpp"
    #define CPUmergeGate mergeGate
    #define CPUproba_state proba_state
    #define CPUSimulator Simulator
#endif

#include "CPUcode/CPUpreprocessor.hpp"
#include "CPUcode/GateMerger.hpp"
#include "CPUcode/simulator.hpp"

#if ((defined __HIPCC__) || (defined __NVCC__))
    #include "HIPcode/HIPpreprocessor.hpp"
    #include "HIPcode/DeviceInfo.hpp"
    #include "HIPcode/GateMerger.hpp"
    #include "HIPcode/GPUMatrix.hpp"
    #include "HIPcode/GPUGate.hpp"
    #include "HIPcode/GPUQuantumCircuit.hpp"
    #include "HIPcode/simulator.hpp"
    
#elif defined _OPENMP

    #warning "openMP is not yet fully supported"
#endif

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