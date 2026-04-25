import launch

if not launch.is_installed("llama-cpp-python"):
    # Install with cuBLAS support for NVIDIA GPU acceleration
    launch.run_pip("install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121", "llama-cpp-python")

if not launch.is_installed("huggingface_hub"):
    launch.run_pip("install huggingface_hub", "huggingface_hub")
