#undef __noinline__
#define __noinline__ noinline
#include <cuda.h>
#include <iostream>

class CudaApp {
    float *devPtr;

public:
    void init(int fd, int bufferSize);
    void step();
};
