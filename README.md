Kate is a project to compute the ouput of quantum computer circuit using tradtionnal GPUs with the backend HIP (compatible with both NVIDIA and AMD)
for now, there is support for multi-GPU but only on 1 node (no MPI)
Speed needs to be measured but shows really good results, especially using special optimizations for preprocessing circuits.

**TO COMPILE**

for HIP

    hipcc main.cpp

for CUDA (hipcc works as well if installed)

    nvcc -x cu main.cpp

Current configuration: It runs a QFT over 24 qbits
Time taken on an RTX 4050 mobile : 1081.77 ms with optimizations and 2927.04 ms without (on QFT, only GateGrouping actually does something)

todo: implement measurment, implement save and load from circuit files, implement new special gates, and clean code along with minor optimizations
