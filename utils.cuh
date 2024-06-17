
// 友好打印错误
void checkCudaError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

// 分配内存空间执行kernel函数
void allocateAndLaunchKernel(
    void *&dev_data,
    const void *host_data,
    size_t count,
    size_t elem_size,
    dim3 blockSize,
    dim3 numBlocks,
    void (*kernel)(void *, int))
{
    checkCudaError(cudaMalloc(&dev_data, count * elem_size));
    checkCudaError(cudaMemcpy(dev_data, host_data, count * elem_size, cudaMemcpyHostToDevice));

    kernel<<<numBlocks, blockSize>>>((decltype(dev_data))dev_data, count);

    cudaFree(dev_data);
}