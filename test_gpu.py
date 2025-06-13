#!/usr/bin/env python3
"""GPU 상태 테스트 스크립트"""

import torch
from ultralytics import YOLO

print("\n" + "="*60)
print("🚀 LitteringDetect2 GPU/CUDA 상태 확인")
print("="*60)

cuda_available = torch.cuda.is_available()
print(f"🔍 CUDA 사용 가능: {'✅ YES' if cuda_available else '❌ NO'}")

if cuda_available:
    try:
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        pytorch_version = torch.__version__
        device_count = torch.cuda.device_count()
        
        print(f"🎮 GPU 장치명: {device_name}")
        print(f"🔧 CUDA 버전: {cuda_version}")
        print(f"🐍 PyTorch 버전: {pytorch_version}")
        print(f"📊 사용 가능한 GPU 수: {device_count}")
        
        # GPU 메모리 정보
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"💾 GPU 메모리: {memory_allocated:.1f}GB 사용 / {total_memory:.1f}GB 전체")
        print(f"⚡ 모델 실행 모드: GPU 가속 모드")
        
    except Exception as e:
        print(f"⚠️  GPU 정보 가져오기 오류: {str(e)}")
        print(f"🔄 CPU 모드로 폴백됩니다")
        cuda_available = False
else:
    print(f"💻 CPU 모드로 실행됩니다")
    print(f"💡 GPU 가속을 위해 CUDA 설치를 권장합니다")

device = torch.device("cuda" if cuda_available else "cpu")
print(f"🎯 최종 사용 장치: {device}")
print("="*60 + "\n")

# YOLO 모델 GPU 테스트
print("🔥"*30)
print("🎯 YOLO 모델 GPU 최적화 시작")
print("🔥"*30)

try:
    model = YOLO('yolov8n.pt')
    
    if cuda_available:
        print("⚡ GPU 사용 가능 - GPU로 모델 이동 중...")
        model.to(device)
        
        print("🔧 모델 최적화 중 (fuse + half precision)...")
        model.model.float()
        model.model.fuse()
        
        if device.type == 'cuda':
            model.model.half()
        
        print("✅ GPU 최적화 완료! (fuse + half precision)")
    else:
        print("💻 CUDA 없음 - CPU 모드로 초기화 중...")
        model.model.cpu().float()
        print("✅ CPU 모드 초기화 완료")

    print(f"🎯 최종 사용 장치: {next(model.model.parameters()).device}")
    print("🔥"*30 + "\n")
    
except Exception as e:
    print(f"❌ 모델 테스트 오류: {str(e)}")

print("✅ GPU 상태 확인 완료!")
