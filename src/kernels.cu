#include "kernels.h"
#include "util.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define get_tid() (blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x)
#define get_bid() (blockIdx.y * gridDim.x + blockIdx.x)

__global__ void changeColors(Vertex* vertexBuffer)
{
    auto tid = get_tid();
    vertexBuffer[tid].color.r += 0.004f;
    if (vertexBuffer[tid].color.r > 1.0f)
        vertexBuffer[tid].color.r = 0.0f;
}

__global__ void rotateVertices(Vertex* vertexBuffer)
{
    auto tid = get_tid();
    auto& pos = vertexBuffer[tid].pos;
    
    glm::mat4 mat = glm::mat4(1.0f);
    mat = glm::rotate(mat, glm::radians(2.0f), glm::vec3(0.0, 0.0, 1.0));
    
    glm::vec4 temp(pos, 0, 1);
    temp = mat * temp;
    pos.x = temp.x;
    pos.y = temp.y;
}

void CudaApp::init()
{
    cudaStreamCreate(&streamToRun);
}

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

void CudaApp::waitOnSemaphore(cudaExternalSemaphore_t& semaphore)
{
    cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;
    memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));
    extSemaphoreWaitParams.params.fence.value = 0;
    extSemaphoreWaitParams.flags = 0;

    cudaWaitExternalSemaphoresAsync(
        &semaphore, &extSemaphoreWaitParams, 1, streamToRun);
}

void CudaApp::signalSemaphore(cudaExternalSemaphore_t& semaphore)
{
    cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
    memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));
    extSemaphoreSignalParams.params.fence.value = 0;
    extSemaphoreSignalParams.flags = 0;

    cudaSignalExternalSemaphoresAsync(
        &semaphore, &extSemaphoreSignalParams, 1, streamToRun);
}

void CudaApp::step()
{
    waitOnSemaphore(vkCudaSemaphore);

    changeColors<<<1, 4, 0, streamToRun>>>(devPtr);

    rotateVertices<<<1, 4, 0, streamToRun>>>(devPtr);

    signalSemaphore(cudaVkSemaphore);
}

void CudaApp::cleanup()
{
    cudaDestroyExternalMemory(cudaExternalMemory);
    cudaDestroyExternalSemaphore(cudaVkSemaphore);
    cudaDestroyExternalSemaphore(vkCudaSemaphore);
}
