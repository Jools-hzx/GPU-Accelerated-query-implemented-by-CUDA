#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data_structures.h"
#include "utils.cuh"

// 基于 GPU 实现多表扫描，聚合和简单计算
__global__ void scanCustomerOrders(Customer *customers, Orders *orders, Lineitem *lineitems, int sizeCust, int sizeOrders, int sizeLineitems, float *customerOrderTotals)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= sizeCust)
        return;

    float total = 0;
    for (int i = 0; i < sizeOrders; i++)
    {
        if (orders[i].O_CUSTKEY == customers[idx].C_CUSTKEY)
        {
            double orderTotal = orders[i].O_TOTALPRICE;
            for (int j = 0; j < sizeLineitems; j++)
            {
                if (lineitems[j].L_ORDERKEY == orders[i].O_ORDERKEY)
                {
                    orderTotal += lineitems[j].L_QUANTITY * 10;
                }
            }
            total += orderTotal;
        }
    }
    customerOrderTotals[idx] = total;
}

void allocateAndLaunchKernel(
    Customer *hostCustomers, int numCustomers,
    Orders *hostOrders, int numOrders,
    Lineitem *hostLineitems, int numLineitems,
    float **hostCustomerOrderTotals,
    dim3 blockSize, dim3 numBlocks)
{
    Customer *devCustomers = nullptr;
    Orders *devOrders = nullptr;
    Lineitem *devLineitems = nullptr;
    float *devCustomerOrderTotals = nullptr;

    checkCudaError(cudaMalloc(&devCustomers, numCustomers * sizeof(Customer)));
    checkCudaError(cudaMemcpy(devCustomers, hostCustomers, numCustomers * sizeof(Customer), cudaMemcpyHostToDevice));

    checkCudaError(cudaMalloc(&devOrders, numOrders * sizeof(Orders)));
    checkCudaError(cudaMemcpy(devOrders, hostOrders, numOrders * sizeof(Orders), cudaMemcpyHostToDevice));

    checkCudaError(cudaMalloc(&devLineitems, numLineitems * sizeof(Lineitem)));
    checkCudaError(cudaMemcpy(devLineitems, hostLineitems, numLineitems * sizeof(Lineitem), cudaMemcpyHostToDevice));

    checkCudaError(cudaMalloc(&devCustomerOrderTotals, numCustomers * sizeof(float)));

    scanCustomerOrders<<<numBlocks, blockSize>>>(devCustomers, devOrders, devLineitems, numCustomers, numOrders, numLineitems, devCustomerOrderTotals);

    checkCudaError(cudaDeviceSynchronize());

    *hostCustomerOrderTotals = new float[numCustomers];
    checkCudaError(cudaMemcpy(*hostCustomerOrderTotals, devCustomerOrderTotals, numCustomers * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(devCustomers);
    cudaFree(devOrders);
    cudaFree(devLineitems);
    cudaFree(devCustomerOrderTotals);
}

int main()
{
    // 模拟数据注入
    // injectData(15000, 150000, 1000);
    injectData(1500000, 150000, 1000);

    // int blockSizes = 512;    //Initial test CUDA and Cpp

    // 测试不同的 blockSize；找到最佳取值
    // 本地 laptop GPU device: NVIDIA GeForce RTX 4060 Laptop GPU
    // 查询设备信息之后的最大支持: Maximum number of threads per thread block:1024
    int blockSizes[] = {128, 256, 512, 1024, 2048};
    float bestTime = FLT_MAX;
    int bestBlockSize = 0;

    for (int size : blockSizes)
    {
        dim3 blockSize(size);
        dim3 numBlocks((customers.size() + blockSize.x - 1) / blockSize.x);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalMilliseconds = 0;
        float *customerOrderTotals = nullptr;

        cudaEventRecord(start);
        allocateAndLaunchKernel(
            customers.data(),
            customers.size(),
            orders.data(),
            orders.size(),
            lineitems.data(),
            lineitems.size(),
            &customerOrderTotals,
            blockSize,
            numBlocks);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // 计算用时
        cudaEventElapsedTime(&totalMilliseconds, start, stop);

        std::cout << "Block Size: " << size << ", Total time taken by GPU: " << totalMilliseconds << " ms\n";

        if (totalMilliseconds < bestTime)
        {
            bestTime = totalMilliseconds;
            bestBlockSize = size;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        delete[] customerOrderTotals;
    }

    std::cout << "Best block size: " << bestBlockSize << " with minimum time: " << bestTime << " ms\n";
    return 0;
}