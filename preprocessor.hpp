#ifndef PREPROCESSORDONE
#define PREPROCESSORDONE

#ifdef __HIPCC__
    #define LOWLEVEL
    #include<hip/hip_runtime.h>
#endif

#ifdef __NVCC__
    #define LOWLEVEL
    #define hipMemcpyDtoH(x, y, z) cudaMemcpy(x, y, z, cudaMemcpyDeviceToHost)
    #define hipMemcpyHtoD(x, y, z) cudaMemcpy(x, y, z, cudaMemcpyHostToDevice)
    #define hipMemcpyDtoHAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyDeviceToHost, w)
    #define hipMemcpyHtoDAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyHostToDevice, w)
    #define hipMalloc cudaMalloc
    #define hipFree cudaFree
    #define hipDeviceSynchronize cudaDeviceSynchronize
    #define hipSetDevice cudaSetDevice
    #define hipDeviceProp_t cudaDeviceProp
    #define hipGetDeviceCount cudaGetDeviceCount
    #define hipDeviceptr_t void*
    #define hipGetDevice cudaGetDevice
    #define hipGetDeviceProperties cudaGetDeviceProperties
    #define hipError_t cudaError_t
    #define hipGetErrorString cudaGetErrorString
    #define hipStream_t cudaStream_t
    #define hipStreamAddCallback cudaStreamAddCallback
    #define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
    #define hipSuccess cudaSuccess
#endif

#ifndef LOWLEVEL
    #define OPENACC_USED
    #define __device__  
    #define __host__  
#else
    #define GPU_CHECK(x)\
err = (x);\
if (err != hipSuccess)\
{\
   	cout << hipGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;\
}
hipError_t err;
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

using namespace std;

#endif