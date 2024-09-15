#undef __noinline__
#define __noinline__ noinline
#include <cuda.h>
#include <iostream>

struct Vertex;

class CudaApp {
    Vertex *devPtr;

public:
    void init(int fd, int bufferSize);
    void step();
};
