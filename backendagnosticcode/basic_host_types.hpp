#ifndef BASICIMPORTED
#define BASICIMPORTED
namespace Kate {

class Complex{
    public:
    double a;
    double b;
    __device__ __host__
    Complex(double a, double b){
        this->a = a;
        this->b = b;
    }
    __device__ __host__
    Complex(double a){
        this->a = a;
        this->b = 0;
    }
    __device__ __host__
    Complex(){
        a = 0;
        b = 0;
    }
    void print(){
        std::cout << a << " + " << b << "i";
    }
    __host__
    double norm(){
        return sqrt(a*a + b*b);
    }
    double angle(){
        return atan(b/a);
    }
    __device__ __host__
    Complex operator+(Complex other){
        return Complex(a+other.a, b+other.b);
    }
    __device__ __host__
    Complex operator-(Complex other){
        return Complex(a-other.a, b-other.b);
    }
    __device__ __host__
    Complex operator*(Complex other){
        return Complex(a*other.a - b*other.b, a*other.b + b*other.a);
    }
    __device__ __host__
    Complex operator*(double other){
        return Complex(a*other, b*other);
    }
    __device__ __host__
    Complex operator*(int other){
        return Complex(a*other, b*other);
    }
    Complex operator/(Complex other){
        double n = other.a*other.a + other.b*other.b;
        return Complex((a*other.a + b*other.b)/n, (other.a*b - a*other.b)/n);
    }
    __device__ __host__
    void operator+=(Complex other){
        a += other.a;
        b += other.b;
    }
    __device__ __host__
    void operator-=(Complex other){
        a -= other.a;
        b -= other.b;
    }
    __device__ __host__
    void operator*=(Complex other){
        double temp = a;
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
    Matrix(){
        n = 0;
        data = NULL;
    }
    Matrix(int n){
        this->n = n;
        data = (T*)malloc(sizeof(T)*n*n);
    }
    Matrix(const Matrix<T>& other){
        n = other.n;
        data = (T*)malloc(sizeof(T)*n*n);
        memcpy(data, other.data, sizeof(T)*n*n);
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
                std::cout << " | ";
            }
            std::cout << std::endl;
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

class GPUGate;

class Gate{
public:
    int identifier = -1; //0 is dense, 2 is H, 3 is CNOT,...
    int optarg;
    double optarg2;
    Complex optarg3;
    Matrix<Complex> densecontent;
    std::vector<int> qbits;
    Gate(int identifier, std::vector<int>& qbits, int optarg = 0, double optarg2 = 0, Complex optarg3 = 0){
        this->identifier = identifier;
        this->optarg = optarg;
        this->optarg2 = optarg2;
        this->optarg3 = optarg3;
        if (identifier == 0) {
            std::cout << "you are creating a dense matrix without specifying it. Memory error incoming" << std::endl;
        }
        if (identifier == 2 && qbits.size() != 1){
            std::cout << "hadamard is on exactly 1 qbit" << std::endl;
        }
        if (identifier == 3 && qbits.size() != 2){
            std::cout << "CNOT is on exactly 2 qbits" << std::endl;
        }
        if (identifier == 4 && qbits.size() != 2){
            std::cout << "Controlled Rk is on exactly 2 qbits" << std::endl;
        }
        if (identifier == 5 && qbits.size() != 3){
            std::cout << "toffoli is on exactly 3 qbits" << std::endl;
        }
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    std::cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << std::endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
    }
    Gate(Matrix<Complex>& densecontent, std::vector<int>& qbits){
        identifier = 0;
        this->densecontent = densecontent;
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    std::cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << std::endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
        if ((1llu << qbits.size()) != densecontent.n){
            std::cout << "size mismatch dense matrix dimension error" << std::endl;
        }
    }
    Gate(int identifier, Matrix<Complex>& densecontent, std::vector<int>& qbits){
        this->identifier = identifier;
        this->densecontent = densecontent;
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    std::cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << std::endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
        if ((1llu << qbits.size()) != densecontent.n){
            std::cout << "size mismatch dense matrix dimension error" << std::endl;
        }
    }
    void print(){
        std::cout << "Identifier : " << identifier << ", Qbits affected : ";
        for (const auto& el: qbits){
            std::cout << el << " ";
        }
        std::cout << std::endl;
    }
};

std::set<int> union_elements(std::set<int>& a, std::set<int>& b){
    std::set<int> c = b;
    for (const auto& el: a){
        c.insert(el);
    }
    return c;
}

std::set<int> union_elements(std::set<int>& a, std::vector<int>& b){
    std::set<int> c = a;
    for (const auto& el: b){
        c.insert(el);
    }
    return c;
}

__device__ Complex exp(Complex i){
    double temp = std::exp(i.a);
    return Complex(temp*cos(i.b), temp*sin(i.b));
}

}
#endif