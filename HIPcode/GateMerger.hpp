#ifndef GateMergerDone
#define GateMergerDone

#include "HIPpreprocessor.hpp"
#include "GPUGate.hpp"
#include "../backendagnosticcode/basic_host_types.hpp"
#include "GPUQuantumCircuit.hpp"
#include "../backendagnosticcode/QuantumCircuit.hpp"

namespace Kate {

__global__
void mergeGatesKernel(GPUGate returnvalue, GPUQuantumCircuit qc, int* coveredqbits_ordered, int gpunumberlog2, int gpuid, int sharedMemMatrixSize){ //the whole circuit is going to merge into a single matrix (that is the plan)
    int bit_to_groupbitnumber[64];
    for (int i = 0; i < qc.nqbits; i++){
        bit_to_groupbitnumber[coveredqbits_ordered[i]] = i;
    }
    
    size_t input = blockIdx.x + (gpuid << (qc.nqbits - gpunumberlog2));
    size_t initline = threadIdx.x;

    extern __shared__ Complex qbitsstateANDmatrixstate[]; //must be of size sizeof(T)*2**nbqbits + sharedMemMatrixSize*sizeof(T)
    Complex* qbitsstate = qbitsstateANDmatrixstate; //size sizeof(T)*2**nbqbits
    Complex* matrixsharedstorage = qbitsstateANDmatrixstate + (1llu << qc.nqbits); //size sharedMemMatrixSize*sizeof(T)

    //initialization
    int work_per_thread0 = (1llu << qc.nqbits)/blockDim.x;
    for (int line = initline*work_per_thread0; line < (initline+1)*work_per_thread0; line++){
        qbitsstate[line] = Complex(((line == input)? 1. : 0.), 0.);
    }
    __syncthreads(); //everyone in the block has fast acces to the whole state, now let s explore the circuit!

    for (int gateid = 0; gateid < qc.gate_number; gateid++){
        qc.gates[gateid].compute(qc.nqbits, qbitsstate, bit_to_groupbitnumber, matrixsharedstorage, sharedMemMatrixSize);
        __syncthreads();
    }
    //in theory we got our input colomn in qbitsstate and we just need to put it in vram at the right place
    for (int line = initline; line < initline+((1llu << qc.nqbits)/blockDim.x); line++){
        returnvalue.densecontent(line, input, qbitsstate[line]);
    }
}

__host__ Gate mergeGate(std::vector<Gate> to_merge, hipStream_t stream = 0){
    //if there is no gpu we will do it on CPU
    int count;
    if (hipGetDeviceCount(&count) != 0){
        //std::cout << "Warning: Merging Could not use HIP/CUDA runtime: Fallback on CPU" << std::endl;
        return CPUmergeGate(to_merge);
    } else if (count == 0){
        //std::cout << "Warning: Merging detected no GPU: Fallback on CPU" << std::endl;
        return CPUmergeGate(to_merge);
    }


    //first we need to not forget about reindexing
    int maxseen = 0;
    std::set<int> total_covered;
    for (const auto& gate: to_merge){
        for (const auto& qbit: gate.qbits){
            total_covered.insert(qbit);
            if (qbit > maxseen) maxseen = qbit;
        }
    }

    int* c_coveredqbits_ordered = (int*)malloc(sizeof(int)*total_covered.size());
    int* coveredqbits_ordered;
    int i = 0;
    for (const auto& el: total_covered){
        c_coveredqbits_ordered[i] = el;
        i++;
    }
    GPU_CHECK(hipMalloc(&coveredqbits_ordered, sizeof(int)*total_covered.size()));
    GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)coveredqbits_ordered, c_coveredqbits_ordered, sizeof(int)*total_covered.size()));
    /*
    //let's build permutation table
    std::vector<int> permuttable(maxseen+1);
    int i = 0;
    for (const auto& el: total_covered){
        permuttable[el] = i;
        i++;
    }
    std::vector<int> temp;
    for (auto& gate: to_merge){
        temp.clear();
        for (const auto& qbit: gate.qbits){
            temp.push_back(permuttable[qbit]);
        }
        gate.qbits = temp;
    }*/
    QuantumCircuit to_mergecirc(to_merge, total_covered.size()); //temporary encapsulation to call the GPU one
    GPUQuantumCircuit gpucircuit = createGPUQuantumCircuit(to_mergecirc);
    //now the circuit is ready for inputing into the kernel
    //let's generate the returned GPUGate
    GPUGate resGPU = createGPUGate(total_covered.size(), std::vector<int>(total_covered.begin(), total_covered.end())); //the kernel will fill the matrix and these informations will be correct
    hipDeviceProp_t devattr;
    int device;
    GPU_CHECK(hipGetDevice(&device));
	GPU_CHECK(hipGetDeviceProperties(&devattr, device));
    size_t totalshared_block = devattr.sharedMemPerBlock/4;
    //only 1 gpu for now. If gpucircuit has less than 5 qbits, we are sad but it should work?
    //ideally, we should do it on CPU when nqbits < 8
    mergeGatesKernel<<<dim3((1llu << gpucircuit.nqbits)), dim3(min((int)1024, (int)(1llu << gpucircuit.nqbits))), totalshared_block, stream>>>(resGPU, gpucircuit, coveredqbits_ordered, 0, 0, (totalshared_block - (1llu << gpucircuit.nqbits))/sizeof(Complex));
    Gate res = createGatefromGPU(resGPU);
    destroyGPUGate(resGPU);
    destroyGPUQuantumCircuit(gpucircuit);
    GPU_CHECK(hipFree(coveredqbits_ordered));
    free(c_coveredqbits_ordered);
    return res;
}

}

#endif