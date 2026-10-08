import torch


def verify_pytorch_installation():
    """
    Verifies that PyTorch is installed and can be imported.
    """
    if torch is None:
        print("PyTorch is not installed. Please install it to proceed.")
        return False
    print("PyTorch is installed. Version:", torch.__version__, flush=True)
    return True

def verify_cuda():
    """
    Verifies that CUDA is available and can be used with PyTorch.
    """
    if torch is None or not torch.cuda.is_available():
        print("CUDA is not available. Check that PyTorch, a compatible GPU, and the correct drivers are installed.", flush=True)
        return False

    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found. Please check your GPU installation.", flush=True)
        return False

    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"CUDA Device {i}: {device_name}")

    print("CUDA verification successful. All devices are available and functioning properly.", flush=True)
    return True

def check_gpu_name():
    """
    Checks the name of the GPU being used by PyTorch.
    """
    if torch is None or not torch.cuda.is_available():
        print("CUDA is not available. Cannot check GPU name.", flush=True)
        return

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    print(f"Current CUDA Device: {device_name}")

def display_vram():
    """
    Displays the total and available VRAM for the current CUDA device.
    """
    if torch is None or not torch.cuda.is_available():
        print("CUDA is not available. Cannot display VRAM.", flush=True)
        return

    device_index = torch.cuda.current_device()
    total_vram = torch.cuda.get_device_properties(device_index).total_memory
    allocated_vram = torch.cuda.memory_allocated(device_index)
    free_vram = total_vram - allocated_vram

    print(f"Total VRAM: {total_vram / (1024 ** 3):.2f} GB")
    print(f"Allocated VRAM: {allocated_vram / (1024 ** 3):.2f} GB")
    print(f"Free VRAM: {free_vram / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    print("-------------- Checking system environment for PyTorch and CUDA --------------", flush=True)
    if verify_pytorch_installation():
        print("\n-------------- Verifying CUDA availability --------------", flush=True)
        if verify_cuda():
            print("\n-------------- Checking GPU name --------------", flush=True)
            check_gpu_name()
            print("\n-------------- Displaying VRAM information --------------", flush=True)
            display_vram()