#ifndef DEVICEINFODONE
#define DEVICEINFODONE

__global__ void testkernel(){
    printf("gpu kernel works\n");
}

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
    cout << "Global memory: " << devattr.totalGlobalMem << " (" << (int)log2(devattr.totalGlobalMem/sizeof(Complex)) << " qbits)" << endl;
    cout << "Shared memory per block : " << devattr.sharedMemPerBlock << " (" << (int)log2(devattr.sharedMemPerBlock/sizeof(Complex)) << " max qbits group execution)" << endl;
    cout << endl;
    testkernel<<<dim3(1), dim3(1)>>>();
    GPU_CHECK(hipDeviceSynchronize());
}

#endif