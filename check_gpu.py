#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU와 CUDA 사용 가능 여부를 확인하는 스크립트
"""

import os
import sys
import platform
import subprocess
import torch
import numpy as np


def print_separator():
    """구분선 출력"""
    print("\n" + "=" * 50 + "\n")


def check_torch_cuda():
    """PyTorch CUDA 가용성 확인"""
    print("PyTorch 버전:", torch.__version__)
    print("CUDA 사용 가능:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA 버전:", torch.version.cuda)
        print("현재 CUDA 장치:", torch.cuda.current_device())
        print("CUDA 장치 수:", torch.cuda.device_count())
        
        # 각 GPU 정보 출력
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print("  이름:", torch.cuda.get_device_name(i))
            print("  메모리 할당량:", torch.cuda.get_device_properties(i).total_memory / 1024**3, "GB")
            print("  멀티프로세서 수:", torch.cuda.get_device_properties(i).multi_processor_count)
            
        # GPU 메모리 정보
        try:
            print("\n현재 GPU 메모리 상태:")
            print("  할당된 메모리:", torch.cuda.memory_allocated() / 1024**3, "GB")
            print("  캐시된 메모리:", torch.cuda.memory_reserved() / 1024**3, "GB")
        except:
            print("GPU 메모리 정보를 가져올 수 없습니다.")
    else:
        print("CUDA를 사용할 수 없습니다.")


def check_system_info():
    """시스템 정보 확인"""
    print("운영체제:", platform.system(), platform.release())
    print("Python 버전:", platform.python_version())
    print("프로세서:", platform.processor())
    
    # NVIDIA-SMI 실행 (Windows)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(["nvidia-smi"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   shell=True)
            if result.returncode == 0:
                print("\nNVIDIA-SMI 출력:")
                print(result.stdout)
            else:
                print("\nNVIDIA-SMI 실행 실패:", result.stderr)
        except Exception as e:
            print(f"\nNVIDIA-SMI 실행 중 오류: {e}")


def test_cuda_computation():
    """간단한 CUDA 연산 테스트"""
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없어 연산 테스트를 건너뜁니다.")
        return
    
    try:
        # CPU에서 텐서 생성
        x_cpu = torch.rand(1000, 1000)
        y_cpu = torch.rand(1000, 1000)
        
        # 시간 측정: CPU 행렬 곱
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        end_time.record()
        torch.cuda.synchronize()
        cpu_time = start_time.elapsed_time(end_time)
        
        # GPU로 텐서 이동
        x_gpu = x_cpu.cuda()
        y_gpu = y_cpu.cuda()
        
        # GPU 캐시 비우기
        torch.cuda.empty_cache()
        
        # 시간 측정: GPU 행렬 곱
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        end_time.record()
        torch.cuda.synchronize()
        gpu_time = start_time.elapsed_time(end_time)
        
        # 결과 비교
        print("\nCUDA 연산 테스트 (1000x1000 행렬 곱):")
        print(f"  CPU 계산 시간: {cpu_time:.2f} ms")
        print(f"  GPU 계산 시간: {gpu_time:.2f} ms")
        print(f"  속도 향상: {cpu_time/gpu_time:.2f}배")
        
        # 결과 검증
        z_gpu_cpu = z_gpu.cpu()
        allclose = torch.allclose(z_cpu, z_gpu_cpu, rtol=1e-5)
        print(f"  결과 일치: {allclose}")
        
    except Exception as e:
        print(f"CUDA 연산 테스트 중 오류 발생: {e}")


def main():
    """메인 함수"""
    print("GPU 및 CUDA 가용성 체크\n")
    
    print_separator()
    print("시스템 정보:")
    check_system_info()
    
    print_separator()
    print("PyTorch 및 CUDA 정보:")
    check_torch_cuda()
    
    print_separator()
    print("CUDA 연산 테스트:")
    test_cuda_computation()
    
    print_separator()
    print("검사 완료")


if __name__ == "__main__":
    main() 