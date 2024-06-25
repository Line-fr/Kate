Kate is a project to compute the ouput of quantum computer circuit using tradtionnal GPUs with the backend HIP or CUDA (coded in HIP with cuda translation via preprocessing)
for now, there is support for multi-GPU but only on 1 node (no MPI)

to compile using hipcc: hipcc main.cpp
to compile using nvcc: nvcc -x cu main.cpp

Kate is a header that you just need to import into the project to use (adapting the compilation steps)

It is currently able to run the qulacs_benchmark over 25 qbits in 3577.15 ms on an RTX 4050 mobile and in 1035.41 ms on an MI200 or in 200ms using 8xMI200x

It uses multiple optimizations using transpilation and memory bandwidth optimizations when running the circuit.
