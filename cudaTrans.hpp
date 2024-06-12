#ifdef __NVCC__
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