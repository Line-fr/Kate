#ifndef BASICIMPORTED
#define BASICIMPORTED

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
__device__ Complex<T> exp(Complex<T> i){
    double temp = exp(i.a);
    return Complex<T>(temp*cos(i.b), temp*sin(i.b));
}

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
class GPUGate;

template<typename T>
class Gate{
public:
    int identifier = -1; //0 is dense, 2 is H, 3 is CNOT,...
    int optarg;
    double optarg2;
    Complex<T> optarg3;
    Matrix<Complex<T>> densecontent;
    vector<int> qbits;
    Gate(int identifier, vector<int>& qbits, int optarg = 0, double optarg2 = 0, Complex<T> optarg3 = 0){
        this->identifier = identifier;
        this->optarg = optarg;
        this->optarg2 = optarg2;
        this->optarg3 = optarg3;
        if (identifier == 0) {
            cout << "you are creating a dense matrix without specifying it. Memory error incoming" << endl;
        }
        if (identifier == 2 && qbits.size() != 1){
            cout << "hadamard is on exactly 1 qbit" << endl;
        }
        if (identifier == 3 && qbits.size() != 2){
            cout << "CNOT is on exactly 2 qbits" << endl;
        }
        if (identifier == 4 && qbits.size() != 2){
            cout << "Controlled Rk is on exactly 2 qbits" << endl;
        }
        if (identifier == 5 && qbits.size() != 3){
            cout << "toffoli is on exactly 3 qbits" << endl;
        }
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
    }
    Gate(Matrix<Complex<T>>& densecontent, vector<int>& qbits){
        identifier = 0;
        this->densecontent = densecontent;
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
        if ((1llu << qbits.size()) != densecontent.n){
            cout << "size mismatch dense matrix dimension error" << endl;
        }
    }
    Gate(int identifier, Matrix<Complex<T>>& densecontent, vector<int>& qbits){
        this->identifier = identifier;
        this->densecontent = densecontent;
        //checking that everyone in qbit is unique
        for (int i = 0; i < qbits.size(); i++){
            for (int j = i+1; j < qbits.size(); j++){
                if (qbits[i] == qbits[j]){
                    cout << "Error while creating a gate: a qbit is present twice: " << qbits[i] << endl;
                    return;
                }
            }
        }
        this->qbits = qbits;
        if ((1llu << qbits.size()) != densecontent.n){
            cout << "size mismatch dense matrix dimension error" << endl;
        }
    }
    Gate(const GPUGate<T>& other){
        identifier = other.identifier;
        densecontent = Matrix<Complex<T>>(other.densecontent);
        qbits.clear();
        int* temp = (int*)malloc(sizeof(int)*other.nbqbits);
        GPU_CHECK(hipMemcpyDtoH(temp, (hipDeviceptr_t)other.qbits, sizeof(int)*other.nbqbits));
        for (int i = 0; i < other.nbqbits; i++){
            qbits.push_back(temp[i]);
        }
        free(temp);
    }
    void print(){
        cout << "Identifier : " << identifier << ", Qbits affected : ";
        for (const auto& el: qbits){
            cout << el << " ";
        }
        cout << endl;
    }
};

set<int> union_elements(set<int>& a, set<int>& b){
    set<int> c = b;
    for (const auto& el: a){
        c.insert(el);
    }
    return c;
}

set<int> union_elements(set<int>& a, vector<int>& b){
    set<int> c = a;
    for (const auto& el: b){
        c.insert(el);
    }
    return c;
}

#endif