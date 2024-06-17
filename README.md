sim_scan_2.cu	      基于CUDA分别扫描三张数据表

sim_scan_2.cpp	    基于CUDA分别扫描三张数据表

scan_agg.cu		  基于 CUDA 扫描三张表+联表查询+运算

scan_agg.cpu		基于 CUDA 扫描三张表+联表查询+运算

utils.cuh		         CUDA 工具代码

data_structures.h 	  定义三张表的列名


## 编译执行
1.  打开命令行界面。 
2.  转到包含的CUDA代码文件 example.cu 的目录.
3.  运行 nvcc 命令并指定输出文件的名称，通常是 .o 结尾的对象文件或直接是可执行文件。例如，要生成可执行文件 example，您可以使用以下命令：
```bash
nvcc -o example example.cu
```

或者，如果您想生成对象文件： 
```bash
nvcc -c -o example.o example.cu
```

4.  如果编译成功，您将在当前目录中看到一个名为 example 的可执行文件或名为 example.o 的对象文件。 
5.  运行可执行文件，如果您是在Linux系统上，使用下面的命令：
```bash
./example
```
 
nvcc 会自动处理CUDA代码的编译，并且调用gcc (或在Windows上是cl.exe) 来编译主程序和处理非CUDA代码。
此外，您可能还需要指定额外的编译选项，例如：

- -arch: 指定计算能力，确保生成的代码能在特定的CUDA架构上运行。
- -G: 启用调试。
- -lineinfo: 保留源行信息。
- -O2 或 -O3: 指定编译器优化级别。
- -use_fast_math: 使用快速数学库。

例如，如果您的GPU支持计算能力5.0，您可以使用以下命令来编译代码：
```bash
nvcc -arch=sm_50 -o example example.cu
```

对于本Demo 可以指定编译选项如下:
```bash
nvcc -arch=sm_89 -o scan_agg_gpu scan_agg.cu -O3
```

然后执行
```bash
./scan_agg_gpu
```
