#ifndef GPUQUANTUMCIRCUITDONE
#define GPUQUANTUMCIRCUITDONE

#include "preprocessor.hpp"
#include "GPUGate.hpp"
#include "basic_host_types.hpp"
#include "GPUMatrix.hpp"
#include "QuantumCircuit.hpp"

class GPUQuantumCircuit{
public:
    int gate_number;
    GPUGate* gates;
    int nqbits;
};

GPUQuantumCircuit createGPUQuantumCircuit(const QuantumCircuit& el){
    GPUQuantumCircuit res;
    res.gate_number = el.gate_set_ordered.size();
    res.nqbits = el.nqbits;
    GPU_CHECK(hipMalloc(&res.gates, sizeof(GPUGate)*res.gate_number));
    GPUGate* temp = (GPUGate*)malloc(sizeof(GPUGate)*res.gate_number);
    for (int i = 0; i < res.gate_number; i++){
        temp[i] = createGPUGate(el.gate_set_ordered[i]);
    }
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)res.gates, temp, sizeof(GPUGate)*res.gate_number));
    free(temp);
    return res;
}

GPUQuantumCircuit createGPUQuantumCircuitAsync(const QuantumCircuit& el){
    GPUQuantumCircuit res;
    res.gate_number = el.gate_set_ordered.size();
    res.nqbits = el.nqbits;
    GPU_CHECK(hipMalloc(&res.gates, sizeof(GPUGate)*res.gate_number));
    GPUGate* temp = (GPUGate*)malloc(sizeof(GPUGate)*res.gate_number);
    for (int i = 0; i < res.gate_number; i++){
        temp[i] = createGPUGateAsync(el.gate_set_ordered[i]);

    }
    GPU_CHECK(hipMemcpyHtoDAsync((hipDeviceptr_t)res.gates, temp, sizeof(GPUGate)*res.gate_number, 0));
    GPU_CHECK(hipStreamAddCallback(0, freecallback, temp, 0));
    return res;
}

void destroyGPUQuantumCircuit(const GPUQuantumCircuit& el){
    GPUGate* temp = (GPUGate*)malloc(sizeof(GPUGate)*el.gate_number);
    GPU_CHECK(hipMemcpyDtoH(temp, (hipDeviceptr_t)el.gates, sizeof(GPUGate)*el.gate_number));
    GPU_CHECK(hipFree(el.gates));
    for (int i = 0; i < el.gate_number; i++){
        destroyGPUGate(temp[i]);
    }
    free(temp);
}

#endif