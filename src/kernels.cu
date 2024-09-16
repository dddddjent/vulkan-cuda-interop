#include "kernels.h"
#include "util.h"

#define get_tid() (blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x)
#define get_bid() (blockIdx.y * gridDim.x + blockIdx.x)

void CudaApp::initSemaphore(int vkCudaFd, int cudaVkFd)
{
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = cudaVkFd;
    externalSemaphoreHandleDesc.flags = 0;
    cudaImportExternalSemaphore(
        &cudaVkSemaphore,
        &externalSemaphoreHandleDesc);

    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = vkCudaFd;
    externalSemaphoreHandleDesc.flags = 0;
    cudaImportExternalSemaphore(
        &vkCudaSemaphore,
        &externalSemaphoreHandleDesc);
}

void CudaApp::initMemHandle(int fd, int bufferSize)
{
    cudaExternalMemoryHandleDesc externalMemoryDesc = {};
    externalMemoryDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryDesc.handle.fd = fd; // File descriptor from Vulkan
    externalMemoryDesc.size = bufferSize;

    cudaImportExternalMemory(&cudaExternalMemory, &externalMemoryDesc);

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = bufferSize;

    void* devPtrVoid;
    cudaExternalMemoryGetMappedBuffer(&devPtrVoid, cudaExternalMemory, &bufferDesc);
    devPtr = reinterpret_cast<Vertex*>(devPtrVoid);
}

__global__ void changeColors(Vertex* vertexBuffer)
{
    auto tid = get_tid();
    vertexBuffer[tid].color.r += 0.004f;
    if (vertexBuffer[tid].color.r > 1.0f)
        vertexBuffer[tid].color.r = 0.0f;
}

void CudaApp::step()
{
    changeColors<<<1, 4>>>(devPtr);
}

void CudaApp::cleanup()
{
    cudaDestroyExternalMemory(cudaExternalMemory);
    cudaDestroyExternalSemaphore(cudaVkSemaphore);
    cudaDestroyExternalSemaphore(vkCudaSemaphore);
}
