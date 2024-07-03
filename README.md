Kate is a project to compute the ouput of quantum computer circuit using tradtionnal GPUs with the backend HIP or CUDA (coded in HIP with cuda translation via preprocessing)
for now, there is support for multi-GPU but only on 1 node (no MPI)

to compile using hipcc: hipcc -O3 main.cpp

to compile using nvcc: nvcc -x cu -O3 main.cpp

to compile for CPU Only: g++ -O3 main.cpp (or clang++ -O3 main.cpp)

to use MPI Kate : mpicxx -O3 main.cpp (beware to include KateMPI.hpp instead of Kate.hpp) Kate MPI only uses CPU


If Kate can't use the GPU, it automatically fallback to the CPU

Note that merging gates will now be done on CPU since merged gates are small

Kate is a header that you just need to import into the project to use (adapting the compilation steps)

Qulacs Benchmarcks 25qbits times:

10086.7 ms on a 7940HS laptop with 6600 MT/s DDR5 RAM (no overclock mode)

3577.15 ms on an RTX 4050 mobile

1035.41 ms on an MI200

200 ms on 8xMI200


It is currently able to run the qulacs_benchmark over 25 qbits in 3577.15 ms on an RTX 4050 mobile and in 1035.41 ms on an MI200 or in 200ms using 8xMI200x

It uses multiple optimizations using transpilation and memory bandwidth optimizations when running the circuit.
