#undef __noinline__
#define __noinline__ noinline
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

struct Vertex;

class CudaApp {
    Vertex* devPtr;
    cudaExternalSemaphore_t cudaVkSemaphore;
    cudaExternalSemaphore_t vkCudaSemaphore;
    cudaExternalMemory_t cudaExternalMemory;
    
public:
    void initSemaphore(int vkCudaFd, int cudaVkFd);
    void initMemHandle(int fd, int bufferSize);
    
    void step();
    void cleanup();
};
