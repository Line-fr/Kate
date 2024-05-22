Kate is a project to compute the ouput of quantum computer circuit using tradtionnal GPUs with the backend HIP (compatible with both NVIDIA and AMD)
for now, there is support for multi-GPU but only on 1 node (no MPI)
Speed needs to be measured but shows relaly good results, especially using special optimizations for preprocessing circuits.

**TO COMPILE**
hipcc main.cpp

todo: implement measurment, implement new special gates, and clean code along with minor optimizations
