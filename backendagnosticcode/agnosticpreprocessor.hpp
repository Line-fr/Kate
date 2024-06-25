#ifndef AGNOPREPROCESSORDONE
#define AGNOPREPROCESSORDONE

#if ! ((defined __NVCC__) || (defined __HIPCC__))
    #define __device__  
    #define __host__  
#endif

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

#define PI 3.1415926535897932384626433
#define SQRT2 1.4142135623730951
#define SQRT2INV 0.7071067811865475
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