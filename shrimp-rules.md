# LitteringDetect2 성능 최적화 개발 가이드라인

## 프로젝트 개요

### 기술 스택
- **언어**: Python 3.x
- **GUI**: PyQt5
- **AI 모델**: YOLO v8 (yolov8n.pt)
- **컴퓨터 비전**: OpenCV
- **딥러닝**: PyTorch, Ultralytics
- **GPU 가속**: CUDA (선택적)

### 핵심 기능
- 실시간 비디오에서 쓰레기 투기 행위 감지
- ROI 기반 영역 제한 처리
- 차량과 쓰레기 객체 간 연관성 분석
- 멀티스레드 비디오 처리 (디코딩/처리/저장 분리)
- 전략 패턴 기반 감지 로직

## 프로젝트 아키텍처

### 핵심 모듈 구조
```
litteringDetect2/
├── litteringDetect.py      # 메인 진입점 및 모델 로딩
├── ui.py                   # PyQt5 GUI 인터페이스
├── processing.py           # 핵심 비디오 처리 로직
├── detection_strategies.py # 감지 전략 클래스들
├── tracker.py              # 객체 추적 알고리즘
├── roi_settings.json       # ROI 설정 저장
├── yolov8n.pt             # YOLO 모델 가중치
└── LOG/                   # 로그 파일 디렉토리
```

### 스레드 아키텍처
- **FrameReaderThread**: 비디오 디코딩 담당
- **VideoThread**: YOLO 추론 및 감지 로직 처리
- **VideoWriterThread**: 이벤트 비디오 저장 담당

## 성능 최적화 규칙

### ⚡ GPU/CUDA 최적화

#### GPU 메모리 관리
- **배치 크기 조정**: `config.batch_size` 값을 GPU 메모리에 맞게 조정
- **메모리 해제**: 모델 추론 후 `torch.cuda.empty_cache()` 호출 필수
- **Half Precision**: GPU 환경에서 `model.half()` 사용으로 메모리 절약

#### 모델 최적화
- **Fuse 적용**: 모델 로딩 시 `model.fuse()` 호출로 레이어 융합
- **장치 일관성**: 모든 텐서를 동일한 장치(CPU/GPU)에서 처리
- **워밍업**: 첫 번째 추론 전 더미 데이터로 모델 워밍업

### 🧵 멀티스레드 최적화

#### 스레드 동기화
- **락 최소화**: `batch_lock`, `buffer_lock`, `event_lock` 사용 시간 최소화
- **큐 크기 제한**: `frame_queue` 크기를 적절히 설정하여 메모리 오버플로우 방지
- **타임아웃 설정**: 모든 큐 작업에 타임아웃 설정

#### 버퍼 관리
- **원형 버퍼**: `deque(maxlen=N)` 사용으로 메모리 사용량 제한
- **동적 크기 조정**: FPS에 따른 버퍼 크기 자동 계산
- **메모리 정리**: 스레드 종료 시 모든 버퍼 명시적 해제

### 🎯 감지 전략 최적화

#### 전략 실행 순서
- **빠른 전략 우선**: `SizeRangeStrategy` → `VehicleOverlapStrategy` → 기타
- **조기 종료**: `detection_logic="ALL"` 모드에서 첫 실패 시 즉시 중단
- **캐싱 활용**: 차량 정보 등 반복 계산 결과 캐싱

#### 성능 모니터링
- **전략별 실행 시간**: 각 전략의 수행 시간 측정 및 로깅
- **메모리 사용량**: 큰 객체들의 메모리 사용량 모니터링
- **FPS 측정**: 실시간 FPS 계산 및 성능 저하 감지

### 📊 로깅 및 디버깅 최적화

#### 프로덕션 모드
- **디버그 로깅 비활성화**: `config.debug_detection = False` 설정
- **로그 레벨 조정**: `logging.WARNING` 이상만 출력
- **파일 로깅**: 중요한 이벤트만 파일에 기록

#### 성능 프로파일링
- **병목 지점 식별**: `cProfile` 또는 `line_profiler` 사용
- **메모리 프로파일링**: `memory_profiler` 로 메모리 누수 확인
- **GPU 모니터링**: `nvidia-smi` 로 GPU 사용률 확인

## 코딩 표준

### 🔧 성능 개선 시 금지사항

#### 절대 수정 금지
- **YOLO 모델 파일**: `yolov8n.pt` 변경 금지
- **핵심 감지 로직**: 정확도에 영향을 주는 임계값 무단 변경 금지
- **ROI 설정**: 사용자 설정 ROI 좌표 임의 변경 금지

#### 신중히 접근
- **전략 비활성화**: 기존 활성화된 전략 비활성화 시 정확도 검증 필수
- **배치 크기 증가**: GPU 메모리 한계 고려하여 점진적 증가
- **스레드 수 증가**: CPU 코어 수와 메모리 용량 고려

### 🎛️ 설정 파라미터 최적화

#### config.py 튜닝 가능 파라미터
```python
# 배치 처리 최적화
config.batch_size = 4  # GPU 메모리에 따라 2-8 범위에서 조정

# 버퍼 크기 최적화
config.buffer_duration = 2  # 이벤트 전 버퍼링 시간(초)
config.post_event_duration = 5  # 이벤트 후 녹화 시간(초)

# 추적 최적화
config.min_frame_count_for_violation = 7  # 위반 판정 최소 프레임 수

# 거리 임계값 최적화
config.distance_trash = 300  # 차량-쓰레기 최대 거리(픽셀)
config.max_vehicle_distance = 200  # 차량 감지 최대 거리

# 디버깅 제어
config.debug_detection = False  # 프로덕션에서는 False
```

### 📁 파일 수정 시 준수사항

#### processing.py 수정 시
- **메모리 누수 방지**: 큰 객체들은 명시적으로 `del` 사용
- **예외 처리**: 모든 스레드에서 예외 발생 시 적절한 정리 작업
- **리소스 해제**: `cap.release()`, `cv2.destroyAllWindows()` 호출 보장

#### detection_strategies.py 수정 시
- **전략 인터페이스 유지**: `DetectionStrategy` 추상 클래스 인터페이스 준수
- **로깅 최소화**: 성능 모드에서는 필수 로그만 출력
- **수치 연산 최적화**: NumPy 배열 연산 활용

#### ui.py 수정 시
- **UI 스레드 분리**: 무거운 작업은 별도 스레드에서 처리
- **시그널/슬롯 사용**: PyQt5의 시그널/슬롯 메커니즘 활용
- **메모리 정리**: 위젯 제거 시 명시적 메모리 해제

## 테스트 및 검증 기준

### 성능 벤치마크
- **최소 FPS**: 실시간 처리 시 최소 15 FPS 유지
- **메모리 사용량**: GPU 메모리 80% 이하 유지
- **CPU 사용률**: 멀티코어 환경에서 70% 이하 유지

### 정확도 검증
- **False Positive**: 오탐 비율 5% 이하 유지
- **False Negative**: 미탐 비율 10% 이하 유지
- **처리 지연**: 실시간 대비 최대 2초 지연 허용

### 테스트 시나리오
- **다양한 해상도**: 720p, 1080p, 4K 영상 테스트
- **다양한 차량 수**: 1대, 3대, 5대 이상 차량 시나리오
- **다양한 조명**: 주간, 야간, 역광 조건 테스트

## AI 에이전트 작업 지침

### 🚀 성능 최적화 작업 순서
1. **현재 성능 측정**: FPS, 메모리 사용량, GPU 활용률 측정
2. **병목 지점 식별**: 프로파일링 도구 사용하여 느린 부분 식별
3. **단계별 최적화**: 한 번에 하나씩 최적화하여 효과 측정
4. **정확도 검증**: 각 최적화 후 감지 정확도 확인
5. **종합 테스트**: 전체 시스템 안정성 및 성능 검증

### ⚠️ 작업 시 주의사항
- **백업 생성**: 주요 파일 수정 전 백업 필수
- **점진적 변경**: 대규모 변경보다는 작은 단위로 개선
- **로그 모니터링**: `LOG/` 디렉토리의 로그 파일 지속 확인
- **Git 커밋**: 각 최적화 단계마다 의미 있는 커밋 메시지

### 🔍 성능 측정 코드 예시
```python
import time
import psutil
import torch

# FPS 측정
start_time = time.time()
frame_count = 0
# ... 프레임 처리 ...
fps = frame_count / (time.time() - start_time)

# GPU 메모리 사용량
if torch.cuda.is_available():
    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB

# CPU 및 시스템 메모리
cpu_percent = psutil.cpu_percent()
memory_percent = psutil.virtual_memory().percent
```

## 최종 검증 체크리스트

### ✅ 성능 요구사항
- [ ] 실시간 처리 속도 (≥15 FPS)
- [ ] GPU 메모리 효율성 (≤80% 사용)
- [ ] CPU 사용률 최적화 (≤70% 사용)
- [ ] 메모리 누수 없음

### ✅ 기능 요구사항  
- [ ] 쓰레기 감지 정확도 유지
- [ ] ROI 설정 정상 동작
- [ ] 비디오 저장 기능 정상
- [ ] GUI 응답성 유지

### ✅ 안정성 요구사항
- [ ] 장시간 실행 시 안정성
- [ ] 예외 상황 처리
- [ ] 리소스 정리 완료
- [ ] 로그 기록 정상