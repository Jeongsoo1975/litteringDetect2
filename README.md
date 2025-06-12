# 쓰레기 무단투기 감지 시스템

## 프로젝트 개요
이 프로젝트는 CCTV 영상에서 쓰레기 무단투기를 감지하고, 차량 번호판을 인식하는 시스템입니다.

## 주요 기능
- 영상에서 쓰레기 무단투기 행위 감지
- 차량 번호판 인식 및 추적
- 결과 로깅 및 시각화

## 설치 방법

### 1. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (Windows)
.\.venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source .venv/bin/activate
```

### 2. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. GPU 지원 (선택사항)
GPU를 활용하려면 다음 명령어로 CUDA 지원 PyTorch를 설치하세요:
```bash
# CUDA 12.1 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
또는 `install_cuda_torch.py` 스크립트를 실행하여 대화형으로 설치하세요:
```bash
python install_cuda_torch.py
```

### 4. 번호판 인식 라이브러리 설정 (선택사항)
번호판 인식 기능을 사용하려면 다음 경로에 TS-ANPR 라이브러리 파일을 배치하세요:
```
bin/windows-x86_64/tsanpr.dll (Windows)
bin/linux-x86_64/libtsanpr.so (Linux)
```

## 사용 방법
프로그램을 실행하려면:
```bash
python litteringDetect.py
```

## 시스템 요구사항
- Python 3.8 이상
- PyQt5
- OpenCV
- PyTorch
- Ultralytics YOLOv8

## 라이센스
이 프로젝트는 MIT 라이센스 하에 배포됩니다. 