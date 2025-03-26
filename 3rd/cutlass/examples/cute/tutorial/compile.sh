# vector add
nvcc -arch=sm_90 -I/root/cutlass/include -I/root/cutlass/tools/util/include vector_add.cu -o vector_add

