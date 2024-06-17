#ifndef DEVICEINFODONE
#define DEVICEINFODONE

__global__ void testkernel(){
    printf("gpu kernel works\n");
}

template<typename T>
void printGpuInfo(){
    int count, device;
    hipDeviceProp_t devattr;
	if (hipGetDeviceCount(&count) != 0){
		cout << "couldnt detect devices, check permissions" << endl;
		return;
	}
    for (int i = 0; i < count; i++){
        GPU_CHECK(hipSetDevice(i));
        GPU_CHECK(hipGetDevice(&device));
	    GPU_CHECK(hipGetDeviceProperties(&devattr, device));
        cout << "GPU " << i << " : " << devattr.name << endl;
    }
    cout << "-----------------------" << endl;
    GPU_CHECK(hipSetDevice(0));
	GPU_CHECK(hipGetDevice(&device));
	GPU_CHECK(hipGetDeviceProperties(&devattr, device));
	cout << "current GPU: " << endl;
	cout << devattr.name << endl;
    cout << endl;
    cout << "Global memory: " << devattr.totalGlobalMem << " (" << (int)log2(devattr.totalGlobalMem/sizeof(T)) << " qbits)" << endl;
    cout << "Shared memory per block : " << devattr.sharedMemPerBlock << " (" << (int)log2(devattr.sharedMemPerBlock/sizeof(T))/2 << " max qbits dense gate execution)" << endl;
    cout << "Registers per blocks : " << devattr.regsPerBlock << " (" << (int)log2(devattr.regsPerBlock/GPUTHREADSNUM) << " max qbits group)" << endl;
	cout << endl;
    testkernel<<<dim3(1), dim3(1)>>>();
    GPU_CHECK(hipDeviceSynchronize());
}

#endif