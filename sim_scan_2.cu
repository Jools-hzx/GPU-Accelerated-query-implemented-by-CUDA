#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data_structures.h"

// 扫描数据行计数器
__device__ int scanCustomerCounter = 0;
__device__ int scanOrderCounter = 0;
__device__ int scanLineitemCounter = 0;

__global__ void scanCustomer(Customer *data, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;

    int localCounter = 0;
    int localKeySum = 0;

    localKeySum += data[idx].C_CUSTKEY;
    localCounter += 1;

    atomicAdd(&scanCustomerCounter, localCounter);
}

__global__ void scanOrders(Orders *data, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;

    int localCounter = 0;
    int localPriceSum = 0;

    localPriceSum += (int)data[idx].O_TOTALPRICE;
    localCounter += 1;

    atomicAdd(&scanOrderCounter, localCounter);
}

__global__ void scanLineitems(Lineitem *data, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;

    int localCounter = 0;
    int localQualitySum = 0;

    localQualitySum += data[idx].L_QUANTITY;
    localCounter += 1;

    atomicAdd(&scanLineitemCounter, localCounter);
}

void checkCudaError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <typename T>
void checkScanResults(const T &deviceCounter, const char *tableName)
{
    int hostCounter;
    // 从设备内存中复制计数器值到主机内存
    checkCudaError(cudaMemcpyFromSymbol(&hostCounter, deviceCounter, sizeof(int), 0, cudaMemcpyDeviceToHost));
    // 输出扫描得到的结果，包括表名
    std::cout << "Total scanned elements [Table-" << tableName << "]: " << hostCounter << std::endl;
}

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

int main()
{
    injectData();

    dim3 blockSize(512);
    dim3 numBlocksCustomers((customers.size() + blockSize.x - 1) / blockSize.x);
    dim3 numBlocksOrders((orders.size() + blockSize.x - 1) / blockSize.x);
    dim3 numBlocksLineitems((lineitems.size() + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalMilliseconds = 0;
    float milliseconds = 0;

    Customer *dev_customers = nullptr;
    Orders *dev_orders = nullptr;
    Lineitem *dev_lineitems = nullptr;

    // Record start time
    cudaEventRecord(start);

    allocateAndLaunchKernel(
        (void *&)dev_customers,
        customers.data(),
        customers.size(),
        sizeof(Customer),
        blockSize,
        numBlocksCustomers,
        (void (*)(void *, int))scanCustomer);

    allocateAndLaunchKernel(
        (void *&)dev_orders,
        orders.data(),
        orders.size(),
        sizeof(Orders),
        blockSize,
        numBlocksOrders,
        (void (*)(void *, int))scanOrders);

    allocateAndLaunchKernel(
        (void *&)dev_lineitems,
        lineitems.data(),
        lineitems.size(),
        sizeof(Lineitem),
        blockSize,
        numBlocksLineitems,
        (void (*)(void *, int))scanLineitems);

    // Record stop time and calculate total time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totalMilliseconds += milliseconds;

    checkScanResults(scanCustomerCounter, "Customer");
    checkScanResults(scanOrderCounter, "Order");
    checkScanResults(scanLineitemCounter, "LineItem");

    std::cout << "[Scan]Total time taken by GPU for all scans: " << totalMilliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}