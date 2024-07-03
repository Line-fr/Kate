#ifndef GPUMATRIXDONE
#define GPUMATRIXDONE

#include "../backendagnosticcode/basic_host_types.hpp"
#include "HIPpreprocessor.hpp"

namespace Kate {

template<typename T>
class GPUMatrix{
public:
    T* data = NULL;
    int n = 0;
    __device__
    T operator()(int a, int b){
        return data[a*n + b];
    }
    __device__
    void operator()(int a, int b, T val){
        data[a*n + b] = val;
    }
};

template<typename T>
GPUMatrix<T> createGPUMatrix(int n){
    GPUMatrix<T> res;
    res.n = n;
    GPU_CHECK(hipMalloc(&(res.data), sizeof(T)*n*n));
    return res;
}

template<typename T>
GPUMatrix<T> createGPUMatrix(const Matrix<T>& other){
    GPUMatrix<T> res = createGPUMatrix<T>(other.n);
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)res.data, other.data, sizeof(T)*other.n*other.n));
    return res;
}

template<typename T>
GPUMatrix<T> createGPUMatrixAsync(const Matrix<T>& other){
    GPUMatrix<T> res = createGPUMatrix<T>(other.n);
    GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)res.data, other.data, sizeof(T)*other.n*other.n, 0));
    return res;
}

template<typename T>
void destroyGPUMatrix(const GPUMatrix<T>& other){
    GPU_CHECK(hipFree(other.data));
}

template<typename T>
Matrix<T> MatrixfromGPUMatrix(const GPUMatrix<T>& other){
    Matrix<T> res;
    res.n = other.n;
    res.data = (T*)malloc(sizeof(T)*other.n*other.n);
    GPU_CHECK(hipMemcpyDtoH(res.data, (hipDeviceptr_t)other.data, sizeof(T)*other.n*other.n));
    return res;
}

}

#endif