import time

import torch


def alocate_tensor_on_gpu():
    """
    Allocates a tensor on the GPU to test if it can be done successfully.
    """
    if torch is None or not torch.cuda.is_available():
        print("CUDA is not available. Cannot allocate tensor on GPU.", flush=True)
        return False

    try:
        # Attempt to allocate a small tensor on the GPU
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        print("Tensor allocated on GPU successfully:", tensor, flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print("Failed to allocate tensor on GPU:", str(e), flush=True)
        return False

def run_matrix_multiplication():
    """
    Performs a simple matrix multiplication on the GPU to test its functionality.
    """
    if torch is None or not torch.cuda.is_available():
        print("CUDA is not available. Cannot perform matrix multiplication on GPU.", flush=True)
        return False

    try:
        # Create two random matrices and perform multiplication on the GPU
        a = torch.zeros((5, 3), device='cuda') # matrix1
        print(f"\nMatrix1 on GPU: {a}")

        b = torch.rand((3, 5), device='cuda') # matrix2
        print(f"\nMatrix2 on GPU: {b}")

        print(f"\n\nmatrix multiplication on GPU: {torch.matmul(a, b)}")  # matrix multiplication
        print("\nMatrix multiplication on GPU completed successfully.", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print("Failed to perform matrix multiplication on GPU:", str(e), flush=True)
        return False

def compare_cpu_vs_gpu_matrix_multiplication():
    """
    Compares the performance of matrix multiplication on CPU vs GPU.
    """
    if torch is None or not torch.cuda.is_available():
        print("CUDA is not available. Cannot compare CPU vs GPU matrix multiplication.", flush=True)
        return False

    try:
        # Create two random matrices
        a_cpu = torch.zeros((5000, 3000))  # matrix1 on CPU
        print(f"\nMatrix1 on CPU: {a_cpu}")

        b_cpu = torch.rand((3000, 5000))  # matrix2 on CPU
        print(f"\nMatrix2 on CPU: {b_cpu}")

        # Perform matrix multiplication on CPU
        
        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        print(f"\nMatrix multiplication on CPU: {c_cpu}")

        cpu_time = time.time() - start_time
        print(f"\nMatrix multiplication on CPU took {cpu_time:.6f} seconds.", flush=True)

        # Move matrices to GPU
        a_gpu = a_cpu.to('cuda') # matrix1 on GPU
        b_gpu = b_cpu.to('cuda') # matrix2 on GPU

        # Perform matrix multiplication on GPU
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        print(f"\nMatrix multiplication on GPU: {c_gpu}")

        gpu_time = time.time() - start_time
        print(f"Matrix multiplication on GPU took {gpu_time:.6f} seconds.", flush=True)

        return True
    except Exception as e:  # noqa: BLE001
        print("Failed to compare CPU vs GPU matrix multiplication:", str(e), flush=True)
        return False

def compare_fp32_vs_fp16_matrix_multiplication():
    """
    Compares the performance of matrix multiplication using FP32 vs FP16 precision on GPU.

    Asynchronous CUDA Clocking: Added torch.cuda.synchronize() before stopping the timers. 
    Because GPU operations are asynchronous in PyTorch, without it code could time how long it took to send the command to the GPU, not how long it took to finish the actual math. 
    This is why FP16 may look artificially slow without warming up!
    """
    if torch is None or not torch.cuda.is_available():
        print("CUDA is not available. Cannot compare FP32 vs FP16 matrix multiplication.", flush=True)
        return False

    try:
        # Create two random matrices in FP32
        a_fp32 = torch.rand((5000, 3000), dtype=torch.float32, device='cuda')  # matrix1 in FP32
        b_fp32 = torch.rand((3000, 5000), dtype=torch.float32, device='cuda')  # matrix2 in FP32

        # Warmup for FP32
        _ = torch.matmul(a_fp32, b_fp32)
        torch.cuda.synchronize()

        # Perform matrix multiplication in FP32
        start_time = time.time()
        c_fp32 = torch.matmul(a_fp32, b_fp32)
        print(f"\nMatrix multiplication in FP32: {c_fp32}")

        fp32_time = time.time() - start_time
        print(f"Matrix multiplication in FP32 took {fp32_time:.6f} seconds.", flush=True)

        # Convert matrices to FP16
        a_fp16 = a_fp32.half()  # matrix1 in FP16
        b_fp16 = b_fp32.half()  # matrix2 in FP16

        # Warmup for FP16
        _ = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()

        # Perform matrix multiplication in FP16
        start_time = time.time()
        c_fp16 = torch.matmul(a_fp16, b_fp16)
        print(f"\nMatrix multiplication in FP16: {c_fp16}")
        
        fp16_time = time.time() - start_time
        print(f"Matrix multiplication in FP16 took {fp16_time:.6f} seconds.", flush=True)

        return True
    except Exception as e:  # noqa: BLE001
        print("Failed to compare FP32 vs FP16 matrix multiplication:", str(e), flush=True)
        return False

if __name__ == "__main__":
    print("Running GPU allocation test...")
    alocate_tensor_on_gpu()
    print("Running matrix multiplication test...")
    run_matrix_multiplication()
    print("Running CPU vs GPU matrix multiplication comparison...")
    compare_cpu_vs_gpu_matrix_multiplication()
    print("Running FP32 vs FP16 matrix multiplication comparison...")
    compare_fp32_vs_fp16_matrix_multiplication()