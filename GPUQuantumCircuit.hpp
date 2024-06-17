#ifndef GPUQUANTUMCIRCUITDONE
#define GPUQUANTUMCIRCUITDONE

#include "preprocessor.hpp"
#include "GPUGate.hpp"
#include "basic_host_types.hpp"
#include "GPUMatrix.hpp"

template<typename T>
class QuantumCircuit;

template<typename T>
class GPUQuantumCircuit{
public:
    int gate_number;
    GPUGate<T>* gates;
    int nqbits;
};

template<typename T>
GPUQuantumCircuit<T> createGPUQuantumCircuit(const QuantumCircuit<T>& el){
    GPUQuantumCircuit<T> res;
    res.gate_number = el.gate_set_ordered.size();
    res.nqbits = el.nqbits;
    GPU_CHECK(hipMalloc(&res.gates, sizeof(GPUGate<T>)*res.gate_number));
    GPUGate<T>* temp = (GPUGate<T>*)malloc(sizeof(GPUGate<T>)*res.gate_number);
    for (int i = 0; i < res.gate_number; i++){
        temp[i] = createGPUGate<T>(el.gate_set_ordered[i]);
    }
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)res.gates, temp, sizeof(GPUGate<T>)*res.gate_number));
    free(temp);
    return res;
}

template<typename T>
GPUQuantumCircuit<T> createGPUQuantumCircuitAsync(const QuantumCircuit<T>& el){
    GPUQuantumCircuit<T> res;
    res.gate_number = el.gate_set_ordered.size();
    res.nqbits = el.nqbits;
    GPU_CHECK(hipMalloc(&res.gates, sizeof(GPUGate<T>)*res.gate_number));
    GPUGate<T>* temp = (GPUGate<T>*)malloc(sizeof(GPUGate<T>)*res.gate_number);
    for (int i = 0; i < res.gate_number; i++){
        temp[i] = createGPUGateAsync<T>(el.gate_set_ordered[i]);

    }
    GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)res.gates, temp, sizeof(GPUGate<T>)*res.gate_number, 0));
    GPU_CHECK(hipStreamAddCallback(0, freecallback, temp, 0));
    return res;
}

template<typename T>
void destroyGPUQuantumCircuit(const GPUQuantumCircuit<T>& el){
    GPUGate<T>* temp = (GPUGate<T>*)malloc(sizeof(GPUGate<T>)*el.gate_number);
    GPU_CHECK(hipMemcpyDtoH(temp, (hipDeviceptr_t)el.gates, sizeof(GPUGate<T>)*el.gate_number));
    GPU_CHECK(hipFree(el.gates));
    for (int i = 0; i < el.gate_number; i++){
        destroyGPUGate<T>(temp[i]);
    }
    free(temp);
}

#endif