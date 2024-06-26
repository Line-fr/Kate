#ifndef CPUDEVICEINFODONE
#define CPUDEVICEINFODONE

namespace Kate{

void printGpuInfo(){
    std::cout << "Kate has been compiled without support for GPU computing" << std::endl;
    std::cout << "You have " << std::thread::hardware_concurrency() << " threads for simulation" << std::endl;
}

}

#endif