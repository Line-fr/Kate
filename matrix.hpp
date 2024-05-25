#include<hip/hip_runtime.h>
#include<stdlib.h>
#include<iostream>
#include<math.h>

#define GPU_CHECK(x)\
err = (x);\
if (err != hipSuccess)\
{\
   	cout << hipGetErrorString(x) << " in " << __FILE__ << " at line " << __LINE__ << endl;\
}
hipError_t err;

using namespace std;

template<typename T>
class Complex{
    public:
    T a;
    T b;
    __device__ __host__
    Complex<T>(T a, T b){
        this->a = a;
        this->b = b;
    }
    __device__ __host__
    Complex<T>(T a){
        this->a = a;
        this->b = 0;
    }
    __device__ __host__
    Complex<T>(){
        a = 0;
        b = 0;
    }
    void print(){
        cout << a << " + " << b << "i";
    }
    __host__
    double norm(){
        return sqrt(a*a + b*b);
    }
    double angle(){
        return atan(b/a);
    }
    __device__ __host__
    Complex<T> operator+(Complex<T> other){
        return Complex<T>(a+other.a, b+other.b);
    }
    __device__ __host__
    Complex<T> operator-(Complex<T> other){
        return Complex<T>(a-other.a, b-other.b);
    }
    __device__ __host__
    Complex<T> operator*(Complex<T> other){
        return Complex<T>(a*other.a - b*other.b, a*other.b + b*other.a);
    }
    __device__ __host__
    Complex<T> operator*(double other){
        return Complex<T>(a*other, b*other);
    }
    __device__ __host__
    Complex<T> operator*(int other){
        return Complex<T>(a*other, b*other);
    }
    Complex<T> operator/(Complex<T> other){
        double n = other.a*other.a + other.b*other.b;
        return Complex<T>((a*other.a + b*other.b)/n, (other.a*b - a*other.b)/n);
    }
    __device__ __host__
    void operator+=(Complex<T> other){
        a += other.a;
        b += other.b;
    }
    __device__ __host__
    void operator-=(Complex<T> other){
        a -= other.a;
        b -= other.b;
    }
    __device__ __host__
    void operator*=(Complex<T> other){
        T temp = a;
        a = a*other.a - b*other.b;
        b = temp*other.b + b*other.a;
    }
    __device__ __host__
    void operator*=(double other){
        a *= other;
        b *= other;
    }
};

template<typename T>
class Matrix;
template<typename T>
class GPUMatrix;

template<typename T>
class Matrix{ //square matrix only
public:
    T* data = NULL;
    int n = 0;
    Matrix(int n){
        this->n = n;
        data = (T*)malloc(sizeof(T)*n*n);
    }
    Matrix(const Matrix<T>& other){
        n = other.n;
        data = (T*)malloc(sizeof(T)*n*n);
        memcpy(data, other.data, sizeof(T)*n*n);
    }
    Matrix(const GPUMatrix<T>& other){
        n = other.n;
        data = (T*)malloc(sizeof(T)*n*n);
        GPU_CHECK(hipMemcpyDtoH(data, (hipDeviceptr_t)other.data, sizeof(T)*n*n));
    }
    ~Matrix(){
        free(data);
    }
    void operator=(const Matrix<T>& other) {
        if (!data) free(data);
        n = other.n;
        data = (T*)malloc(sizeof(T)*n*n);
        memcpy(data, other.data, sizeof(T)*n*n);
    }
    void copy(Matrix<T>& other){
        memcpy(data, other.data, sizeof(T)*n*n);
    }
    void fill(T val){
        for (int i = 0; i < n*n; i++){
            data[i] = val;
        }
    }
    void print(){
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                data[i*n+j].print();
                cout << " | ";
            }
            cout << endl;
        }
    }
    T operator()(int a, int b){
        return data[a*n + b];
    }
    void operator()(int a, int b, T val){
        data[a*n + b] = val;
    }
    Matrix<T> operator+(Matrix<T>& other){
        Matrix<T> result(n);
        for (int i = 0; i < n*n; i++){
            result.data[i] = other.data[i] + data[i];
        }
        return result;
    }
    Matrix<T> operator-(Matrix<T>& other){
        Matrix<T> result(n);
        for (int i = 0; i < n*n; i++){
            result.data[i] = other.data[i] - data[i];
        }
        return result;
    }
    void operator+=(Matrix<T>& other){
        for (int i = 0; i < n*n; i++){
            data[i] += other.data[i];
        }
    }
    void operator-=(Matrix<T>& other){
        for (int i = 0; i < n*n; i++){
            data[i] -= other.data[i];
        }
    }
    Matrix<T> operator*(Matrix<T>& other){
        Matrix<T> result(n);
        T sum;
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                sum = 0;
                for (int k = 0; k < n; k++){
                    sum += data[i*n + k]*other(k, j);
                }
                result(i, j, sum);
            }
        }
        return result;
    }
    void operator*=(double other){
        for (int i = 0; i < n*n; i++){
            data[i] *= other;
        }
    }
    
};

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

//only accept matrix up to 16 qbits. If needed another implemention could allow getting way higher
template<typename T>
__global__ void matFillKernel(GPUMatrix<T> mat, T val){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    mat.data[i] = val;
}

template<typename T>
__host__ void matFill(GPUMatrix<T>& mat, T val, hipStream_t stream = 0){
    int n = mat.n;
    if (n >= 32){
        matFillKernel<<<dim3(n*n/1024), dim3(1024), 0, stream>>>(mat, val);
    } else {
        matFillKernel<<<dim3(1), dim3(n*n), 0, stream>>>(mat, val);
    }
}

//only accept matrix up to 16 qbits. If needed another implemention could allow getting way higher
template<typename T>
__global__ void matAddKernel(GPUMatrix<T> a, GPUMatrix<T> b, GPUMatrix<T> c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c.data[i] = a.data[i] + b.data[i];
}

template<typename T>
__host__ GPUMatrix<T> matAdd(GPUMatrix<T>& a, GPUMatrix<T>& b, hipStream_t stream = 0){
    int n = a.n;
    GPUMatrix<T> c = createGPUMatrix<T>(n);
    if (n >= 32){
        matAddKernel<<<dim3(n*n/1024), dim3(1024), 0, stream>>>(a, b, c);
    } else {
        matAddKernel<<<dim3(1), dim3(n*n), 0, stream>>>(a, b, c);
    }
    return c;
}

template<typename T>
__global__ void matSelfAddKernel(GPUMatrix<T> a, GPUMatrix<T> b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    a.data[i] += b.data[i];
}

template<typename T>
__host__ void matSelfAdd(GPUMatrix<T>& a, GPUMatrix<T>& b, hipStream_t stream = 0){
    int n = a.n;
    if (n >= 32){
        matSelfAddKernel<<<dim3(n*n/1024), dim3(1024), 0, stream>>>(a, b);
    } else {
        matSelfAddKernel<<<dim3(1), dim3(n*n), 0, stream>>>(a, b);
    }
}

//only accept matrix up to 16 qbits. If needed another implemention could allow getting way higher
template<typename T>
__global__ void matMinusKernel(GPUMatrix<T> a, GPUMatrix<T> b, GPUMatrix<T> c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c.data[i] = a.data[i] - b.data[i];
}

template<typename T>
__host__ GPUMatrix<T> matMinus(GPUMatrix<T>& a, GPUMatrix<T>& b, hipStream_t stream = 0){
    int n = a.n;
    GPUMatrix<T> c = createGPUMatrix<T>(n);
    if (n >= 32){
        matMinusKernel<<<dim3(n*n/1024), dim3(1024), 0, stream>>>(a, b, c);
    } else {
        matMinusKernel<<<dim3(1), dim3(n*n), 0, stream>>>(a, b, c);
    }
    return c;
}

template<typename T>
__global__ void matSelfMinusKernel(GPUMatrix<T> a, GPUMatrix<T> b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    a.data[i] -= b.data[i];
}

template<typename T>
__host__ void matSelfMinus(GPUMatrix<T>& a, GPUMatrix<T>& b, hipStream_t stream = 0){
    int n = a.n;
    if (n >= 32){
        matSelfMinusKernel<<<dim3(n*n/1024), dim3(1024), 0, stream>>>(a, b);
    } else {
        matSelfMinusKernel<<<dim3(1), dim3(n*n), 0, stream>>>(a, b);
    }
}

template<typename T>
__global__ void matSelfProdDoubleKernel(GPUMatrix<T> a, double b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    a.data[i] *= b;
}

template<typename T>
__host__ void matSelfProdDouble(GPUMatrix<T>& a, double b, hipStream_t stream = 0){
    int n = a.n;
    if (n >= 32){
        matSelfProdDoubleKernel<<<dim3(n*n/1024), dim3(1024), 0, stream>>>(a, b);
    } else {
        matSelfProdDoubleKernel<<<dim3(1), dim3(n*n), 0, stream>>>(a, b);
    }
}

template<typename T>
__global__ void matSelfProdDoubleKernel(GPUMatrix<T> a, int b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    a.data[i] *= b;
}

template<typename T>
__host__ void matSelfProdDouble(GPUMatrix<T>& a, int b, hipStream_t stream = 0){
    int n = a.n;
    if (n >= 32){
        matSelfProdDoubleKernel<<<dim3(n*n/1024), dim3(1024), 0, stream>>>(a, b);
    } else {
        matSelfProdDoubleKernel<<<dim3(1), dim3(n*n), 0, stream>>>(a, b);
    }
}

//support from 5 qbits to 16 qbits
template<typename T>
__global__ void matProdKernelShared(GPUMatrix<T> a, GPUMatrix<T> b, GPUMatrix<T> c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    T sum = 0.;

    extern __shared__ T Ad[];
    extern __shared__ T Bd[];

    for (int k = 0; k < a.n; k += 32){
        Ad[threadIdx.x*32+threadIdx.y] = a(i, k+j);
        Bd[threadIdx.x*32+threadIdx.y] = b(k+i, j);
        __syncthreads();
        for (int l = 0; l < 32; l++){
            sum += Ad[threadIdx.x*32+l] * Bd[l*32+threadIdx.y];
        }
        __syncthreads();
    }
    c(i, j, sum);
}

//support from 0qbits to 16qbits
template<typename T>
__global__ void smallMatProdKernel(GPUMatrix<T> a, GPUMatrix<T> b, GPUMatrix<T> c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    T sum = 0.;
    for (int k = 0; k < a.n; k++){
        sum += a(i, k)*b(k, j);
    }
    c(i, j, sum);
}

//in general using the gpu for small things is not a so good idea. But if it avoids data transfer, still worth it to implement
template<typename T>
__host__ GPUMatrix<T> matProd(GPUMatrix<T>& a, GPUMatrix<T>& b, hipStream_t stream = 0){
    int n = a.n;
    GPUMatrix<T> c = createGPUMatrix<T>(n);
    if (n >= 32){
        smallMatProdKernel<<<dim3(n/32, n/32), dim3(32, 32), sizeof(T)*1024, stream>>>(a, b, c);
    } else {
        smallMatProdKernel<<<dim3(1, 1), dim3(n, n), 0, stream>>>(a, b, c);
    }
    return c;
}

