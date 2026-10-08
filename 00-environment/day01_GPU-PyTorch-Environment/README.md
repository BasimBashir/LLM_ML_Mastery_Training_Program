My Hardware is Intel i5 12600F processor with 64GB DDR4 memory and Nvidia RTX 3090 with 24GB VRAM.

I have installed Pytorch version: 2.14.1+cu126

I have single 3090 GPU so my GPU Device ID shows 0.

a Dot product or matrix multiplication on my machine shows these values:
matrix_A = rand(5000, 3000)

matrix_B = rand(3000, 5000)

Rule: Number of rows of second matrix should be equal to the number of columns of first matrix. 

matrix is shown as: M = (rows, columns)

cpu time = 0.339017 seconds

gpu time = 0.011384 seconds

So, we can clearly see that gpu will win to process a dot product from cpu quite easily but what about two matrices but with different datatypes on same gpu?

So, take same matrices as above and calculate time based on datatypes.

FP32 processing time on GPU: 0.011688 seconds

FP16 processing time on GPU: 0.004662 seconds


But always note that without warming up the gpu FP32 will use less processing time because the GPU asynchronously do the calculation so always warm up a gpu first before doing fp16 or lower calculations using `torch.cuda.synchronize()` and a dummy dot product for warming up.


Watch `system_check.py` and `benchmark_gpu.py` for implementation details, and
see `LEARNING_LOG.md` for the recorded environment and experiment notes.
