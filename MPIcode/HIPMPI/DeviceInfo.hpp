#ifndef MPIDEVICEINFODONE
#define MPIDEVICEINFODONE

__global__ void testkernel(){
    printf("gpu kernel works\n");
}

namespace Kate{
    void printGpuInfo(MPI_Comm comm = MPI_COMM_WORLD){
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        
        if (rank == 0){
            std::cout << "Compiled with MPI with GPU support. You have " << size << " MPI processes right now" << std::endl;
            std::cout << "CPU Info: " << std::thread::hardware_concurrency() << " threads" << std::endl;
            int count, device;
            hipDeviceProp_t devattr;
            if (hipGetDeviceCount(&count) != 0){
                std::cout << "couldnt detect devices, check permissions" << std::endl;
                return;
            }
            for (int i = 0; i < count; i++){
                GPU_CHECK(hipSetDevice(i));
                GPU_CHECK(hipGetDevice(&device));
                GPU_CHECK(hipGetDeviceProperties(&devattr, device));
                std::cout << "GPU " << i << " : " << devattr.name << std::endl;
            }
            std::cout << "-----------------------" << std::endl;
            GPU_CHECK(hipSetDevice(0));
            GPU_CHECK(hipGetDevice(&device));
            GPU_CHECK(hipGetDeviceProperties(&devattr, device));
            std::cout << "current GPU: " << std::endl;
            std::cout << devattr.name << std::endl;
            std::cout << std::endl;
            std::cout << "Global memory: " << devattr.totalGlobalMem << " (" << (int)log2(devattr.totalGlobalMem/sizeof(Complex)) << " qbits)" << std::endl;
            std::cout << "Shared memory per block : " << devattr.sharedMemPerBlock << " (" << (int)log2(devattr.sharedMemPerBlock/sizeof(Complex)) << " max qbits group execution)" << std::endl;
            std::cout << std::endl;
            testkernel<<<dim3(1), dim3(1)>>>();
            GPU_CHECK(hipDeviceSynchronize());
        }
    }
}

#endif