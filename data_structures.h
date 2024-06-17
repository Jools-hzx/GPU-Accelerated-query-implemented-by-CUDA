#pragma once
#include <vector>
#include <cstdio>
#include <iostream>
#include <cmath>

// 定义各个表的列名
struct Customer
{
    int C_CUSTKEY;
    char C_NAME[100];
};

struct Orders
{
    int O_CUSTKEY;
    int O_ORDERKEY;
    char O_ORDERDATE[10];
    double O_TOTALPRICE;
};

struct Lineitem
{
    int L_ORDERKEY;
    int L_QUANTITY;
};

std::vector<Customer> customers;
std::vector<Orders> orders;
std::vector<Lineitem> lineitems;

template <typename T>
void generateRandomData(std::vector<T> &vec, size_t numElements);

// 生成测试数据，每一张表插入 10000 条数据
template <>
void generateRandomData(std::vector<Customer> &vec, size_t numElements)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        Customer customer;
        customer.C_CUSTKEY = i % 1000;
        snprintf(customer.C_NAME, sizeof(customer.C_NAME), "Customer #%zu", i);
        vec.push_back(customer);
    }
    // 打印生成数据的数量
    std::cout << "Generated " << numElements << " elements for Customers table" << std::endl;
}

template <>
void generateRandomData(std::vector<Orders> &vec, size_t numElements)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        Orders order;
        order.O_CUSTKEY = i % 100;
        order.O_ORDERKEY = i;
        snprintf(order.O_ORDERDATE, sizeof(order.O_ORDERDATE), "2023-01-%02zu", (i % 30) + 1);
        order.O_TOTALPRICE = std::fmod(200.0 * i, 1000.0);
        vec.push_back(order);
    }
    // 打印生成数据的数量
    std::cout << "Generated " << numElements << " elements for Orders table" << std::endl;
}

template <>
void generateRandomData(std::vector<Lineitem> &vec, size_t numElements)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        Lineitem item;
        item.L_ORDERKEY = i % 100;
        item.L_QUANTITY = i % 5 + 1 % 100;
        vec.push_back(item);
    }
    // 打印生成数据的数量
    std::cout << "Generated " << numElements << " elements for Lineitems table" << std::endl;
}

void injectData()
{

    // 假设 generateRandomData 是一个现有的函数，用来生成随机数据。
    generateRandomData(customers, 1500000);
    generateRandomData(orders, 15000000);
    generateRandomData(lineitems, 59986052);
}

// 可以指定各个表插入数据
void injectData(int numCust, int numOrder, int lineItmesNum)
{

    // 假设 generateRandomData 是一个现有的函数，用来生成随机数据。
    generateRandomData(customers, numCust);
    generateRandomData(orders, numOrder);
    generateRandomData(lineitems, lineItmesNum);
}