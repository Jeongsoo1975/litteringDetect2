"""
CUDA 지원 PyTorch 설치 스크립트

이 스크립트는 현재 Python 환경에 CUDA 지원 PyTorch를 설치합니다.
스크립트를 실행하기 전에 가상환경이 활성화되어 있는지 확인하세요.
"""

import os
import sys
import platform
import subprocess

def install_torch_cuda():
    """CUDA 지원 PyTorch 설치"""
    print("CUDA 지원 PyTorch 설치를 시작합니다...")
    
    # CUDA 버전에 따른 설치 명령어
    cuda_version = input("""
설치할 CUDA 버전을 선택하세요 (1-3):
1. CUDA 11.8 (기본)
2. CUDA 12.1
3. CPU 전용 (CUDA 미사용)

선택: """)
    
    if cuda_version == "2":
        # CUDA 12.1 버전
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif cuda_version == "3":
        # CPU 전용
        cmd = "pip install torch torchvision torchaudio"
    else:
        # 기본: CUDA 11.8
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    print(f"\n실행할 명령어: {cmd}\n")
    
    # 사용자 확인
    confirm = input("계속 진행하시겠습니까? (y/n): ")
    if confirm.lower() != 'y':
        print("설치가 취소되었습니다.")
        return
    
    # 명령어 실행
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print("\nPyTorch 설치가 완료되었습니다.")
        
        # 설치 확인
        print("\n설치된 PyTorch 버전 확인 중...")
        subprocess.run([sys.executable, "-c", 
                      "import torch; print('PyTorch 버전:', torch.__version__); "
                      "print('CUDA 사용 가능:', torch.cuda.is_available()); "
                      "print('CUDA 버전:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"],
                     shell=True)
        
        # 추가 패키지 설치 여부
        install_more = input("\n추가 패키지(numpy, matplotlib, pandas)를 설치하시겠습니까? (y/n): ")
        if install_more.lower() == 'y':
            subprocess.run("pip install numpy matplotlib pandas", shell=True, check=True)
            print("추가 패키지 설치가 완료되었습니다.")
        
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")

if __name__ == "__main__":
    # 가상환경 확인
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("경고: 가상환경이 활성화되지 않은 것 같습니다.")
        continue_anyway = input("계속 진행하시겠습니까? (y/n): ")
        if continue_anyway.lower() != 'y':
            print("설치가 취소되었습니다. 가상환경을 활성화한 후 다시 시도하세요.")
            sys.exit(1)
    
    install_torch_cuda() 