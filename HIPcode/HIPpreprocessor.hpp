#ifndef HIPpreprocessingDone
#define HIPpreprocessingDone

#ifdef __HIPCC__
    #include<hip/hip_runtime.h>
#elif defined __NVCC__
    #define LOWLEVEL
    #define hipMemcpyDtoH(x, y, z) cudaMemcpy(x, y, z, cudaMemcpyDeviceToHost)
    #define hipMemcpyHtoD(x, y, z) cudaMemcpy(x, y, z, cudaMemcpyHostToDevice)
    #define hipMemcpyDtoHAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyDeviceToHost, w)
    #define hipMemcpyHtoDAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyHostToDevice, w)
    #define hipMemcpyPeer cudaMemcpyPeer
    #define hipMemcpyPeerAsync cudaMemcpyPeerAsync
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


hipError_t errhip;

#define GPU_CHECK(x)\
errhip = (x);\
if (errhip != hipSuccess)\
{\
   	std::cout << hipGetErrorString(errhip) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;\
}

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

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#endif