import torch

# 1. 설치된 PyTorch 버전 확인
print(f"PyTorch Version: {torch.__version__}")

# 2. PyTorch가 사용하는 CUDA 버전 확인
print(f"PyTorch's CUDA Version: {torch.version.cuda}")

# 3. 현재 시스템에서 CUDA 사용 가능 여부 확인
is_available = torch.cuda.is_available()
print(f"CUDA is available: {is_available}")

if is_available:
    # 4. 현재 사용 중인 GPU의 CUDA 컴퓨팅 성능 확인
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")