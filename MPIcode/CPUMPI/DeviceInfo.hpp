#ifndef MPIDEVICEINFODONE
#define MPIDEVICEINFODONE

namespace Kate{
    void printGpuInfo(MPI_Comm comm = MPI_COMM_WORLD){
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        
        if (rank == 0){
            std::cout << "Compiled with MPI. You have " << size << " MPI processes right now" << std::endl;
            std::cout << "Rank " << rank << " runs with " << std::thread::hardware_concurrency() << " Threads" << std::endl;
        }
    }
}

#endif