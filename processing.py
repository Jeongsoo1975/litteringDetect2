#process.py
import sys
import os
import cv2
import json
import math
import time
import torch
import logging
import traceback
import threading
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QSlider, QGroupBox, QFormLayout, QMessageBox, QSpinBox, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap  # 이 줄 추가
import queue
from detection_strategies import DetectionStrategyManager, VehicleOverlapStrategy, SizeRangeStrategy, VehicleDistanceStrategy, GravityDirectionStrategy, DirectionAlignmentStrategy, VehicleAssociationStrategy
import csv  # csv 모듈 추가
from high_performance_cache import HighPerformanceCacheManager, create_high_performance_cache

# 싱글톤 모델 인스턴스를 저장할 전역 변수
_model_instance = None
logging.getLogger("ultralytics").setLevel(logging.WARNING)

##########################
# 스피너(Spinner) 및 모델 로딩 래퍼
##########################
def spinner_task(stop_event):
    spinner_chars = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r모델 로딩 중... {spinner_chars[idx]}")
        sys.stdout.flush()
        idx = (idx + 1) % len(spinner_chars)
        time.sleep(0.1)

    sys.stdout.write("\r모델 로딩 완료!     \n")
    sys.stdout.flush()


def load_model_with_spinner(model_path="yolov8n.pt"):
    """
    스피너를 표시하면서 모델을 로드하는 함수
    싱글톤 패턴을 적용하여 모델이 한 번만 로드되도록 함
    """
    global _model_instance

    # 이미 모델이 로드되어 있으면 그것을 반환
    if _model_instance is not None:
        return _model_instance

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner_task, args=(stop_event,))
    spinner_thread.start()

    try:
        _model_instance = YOLO(model_path)
    except Exception as e:
        logging.error(f"모델 로딩 중 오류 발생: {str(e)}")
        stop_event.set()
        spinner_thread.join()
        raise

    stop_event.set()
    spinner_thread.join()
    return _model_instance


##########################
# 설정 파일 관리 함수들 (안전한 로깅 포함)
##########################
def load_settings_from_file(file_path="default_settings.txt", timeout=5):
    """
    타임아웃 및 안전 처리가 추가된 설정 파일 로딩 함수
    
    Args:
        file_path: 설정 파일 경로
        timeout: 파일 읽기 타임아웃 (초)
    
    Returns:
        dict: 설정값 딕셔너리
    """
    settings = {}
    
    try:
        # 파일 존재 확인
        if not os.path.exists(file_path):
            try:
                logger.warning(f"설정 파일이 없습니다: {file_path}. 기본값을 사용합니다.")
            except NameError:
                print(f"경고: 설정 파일이 없습니다: {file_path}. 기본값을 사용합니다.")
            return {}
        
        # 파일 크기 확인 (너무 크면 문제가 있을 수 있음)
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024:  # 10KB 초과
            try:
                logger.warning(f"설정 파일이 너무 큽니다: {file_size} bytes. 최대 10KB까지만 지원됩니다.")
            except NameError:
                print(f"경고: 설정 파일이 너무 큽니다: {file_size} bytes. 최대 10KB까지만 지원됩니다.")
            return {}
        
        # 타임아웃을 사용한 파일 읽기
        result = {}
        exception_occurred = threading.Event()
        
        def read_file():
            nonlocal result, exception_occurred
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 파일 내용이 비어있는지 확인
                if not content.strip():
                    try:
                        logger.warning(f"설정 파일이 비어있습니다: {file_path}")
                    except NameError:
                        print(f"경고: 설정 파일이 비어있습니다: {file_path}")
                    return
                
                # 줄 단위로 파싱 (최대 100줄까지만 처리)
                lines = content.split('\n')
                if len(lines) > 100:
                    try:
                        logger.warning(f"설정 파일의 줄 수가 너무 많습니다: {len(lines)}줄. 최대 100줄까지만 처리됩니다.")
                    except NameError:
                        print(f"경고: 설정 파일의 줄 수가 너무 많습니다: {len(lines)}줄. 최대 100줄까지만 처리됩니다.")
                    lines = lines[:100]
                
                parsed_settings = {}
                line_count = 0
                
                for line in lines:
                    line_count += 1
                    line = line.strip()
                    
                    # 주석과 빈 줄 건너뛰기
                    if not line or line.startswith('#'):
                        continue
                    
                    # key=value 형식 파싱
                    if '=' in line:
                        try:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # 빈 키나 값은 무시
                            if not key or not value:
                                continue
                            
                            # 데이터 타입 변환
                            if value.lower() in ['true', 'false']:
                                parsed_settings[key] = value.lower() == 'true'
                            elif value.replace('.', '', 1).replace('-', '', 1).isdigit():  # 숫자 (int 또는 float, 음수 포함)
                                parsed_settings[key] = float(value) if '.' in value else int(value)
                            else:
                                parsed_settings[key] = value
                        except ValueError as ve:
                            try:
                                logger.warning(f"설정 파일의 {line_count}줄에서 값 변환 오류: {ve}")
                            except NameError:
                                print(f"경고: 설정 파일의 {line_count}줄에서 값 변환 오류: {ve}")
                            continue
                
                result = parsed_settings
                
            except UnicodeDecodeError:
                try:
                    logger.error(f"설정 파일 인코딩 문제: {file_path}. UTF-8로 저장해주세요.")
                except NameError:
                    print(f"오류: 설정 파일 인코딩 문제: {file_path}. UTF-8로 저장해주세요.")
                exception_occurred.set()
            except PermissionError:
                try:
                    logger.error(f"설정 파일 접근 권한 없음: {file_path}")
                except NameError:
                    print(f"오류: 설정 파일 접근 권한 없음: {file_path}")
                exception_occurred.set()
            except Exception as e:
                try:
                    logger.error(f"설정 파일 읽기 중 예외 발생: {str(e)}")
                except NameError:
                    print(f"오류: 설정 파일 읽기 중 예외 발생: {str(e)}")
                exception_occurred.set()
        
        # 별도 스레드에서 파일 읽기 실행
        read_thread = threading.Thread(target=read_file, daemon=True)
        read_thread.start()
        read_thread.join(timeout=timeout)
        
        # 타임아웃 체크
        if read_thread.is_alive():
            try:
                logger.error(f"설정 파일 읽기 타임아웃 ({timeout}초). 기본값을 사용합니다.")
            except NameError:
                print(f"오류: 설정 파일 읽기 타임아웃 ({timeout}초). 기본값을 사용합니다.")
            return {}
        
        # 예외 발생 체크
        if exception_occurred.is_set():
            try:
                logger.error(f"설정 파일 읽기 중 예외 발생. 기본값을 사용합니다.")
            except NameError:
                print(f"오류: 설정 파일 읽기 중 예외 발생. 기본값을 사용합니다.")
            return {}
        
        # 성공적으로 읽은 경우
        if result:
            try:
                logger.info(f"설정 파일 로드 성공: {file_path}")
                logger.info(f"로드된 설정값: {result}")
            except NameError:
                print(f"설정 파일 로드 성공: {file_path}")
                print(f"로드된 설정값: {result}")
        else:
            try:
                logger.warning(f"설정 파일에서 유효한 설정을 찾을 수 없습니다: {file_path}")
            except NameError:
                print(f"경고: 설정 파일에서 유효한 설정을 찾을 수 없습니다: {file_path}")
        
        return result
        
    except Exception as e:
        try:
            logger.error(f"설정 파일 로드 중 예외 발생: {str(e)}")
        except NameError:
            print(f"오류: 설정 파일 로드 중 예외 발생: {str(e)}")
        return {}

def save_settings_to_file(config, file_path="default_settings.txt"):
    """현재 설정값을 파일에 저장하는 함수"""
    try:
        content = f"""# LitteringDetect2 기본 설정값
# 이 파일은 프로그램 시작 시 자동으로 로드됩니다.
# 각 설정값을 수정한 후 프로그램을 재시작하면 새로운 값이 적용됩니다.

# 객체 크기 설정 (픽셀 단위)
min_size={config.min_size}
max_size={config.max_size}

# YOLO 설정
yolo_confidence_value={config.yolo_confidence_value}
batch_size={config.batch_size}

# 거리 설정 (픽셀 단위)
distance_trash={config.distance_trash}
gravity_direction_threshold={config.gravity_direction_threshold}
horizontal_direction_threshold={config.horizontal_direction_threshold}
max_vehicle_distance={config.max_vehicle_distance}

# 차량 겹침 임계값 (비율, 0.0-1.0)
vehicle_overlap_threshold={config.vehicle_overlap_threshold}

# 위반 판정을 위한 최소 프레임 카운트
min_frame_count_for_violation={config.min_frame_count_for_violation}

# 감지 로직 ("ANY": 어느 하나라도 충족, "ALL": 모두 충족)
detection_logic={config.detection_logic}

# 디버깅 설정 (true/false)
debug_detection={str(config.debug_detection).lower()}

# 성능 모니터링 설정
performance_monitoring={str(config.performance_monitoring).lower()}
target_fps={config.target_fps}
max_gpu_memory_percent={config.max_gpu_memory_percent}
max_cpu_percent={config.max_cpu_percent}
performance_warning_threshold={config.performance_warning_threshold}
performance_log_interval={config.performance_log_interval}

# 출력 디렉토리
output_dir={config.output_dir}
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        try:
            logger.info(f"설정 파일 저장 완료: {file_path}")
        except NameError:
            print(f"설정 파일 저장 완료: {file_path}")
        
    except Exception as e:
        try:
            logger.error(f"설정 파일 저장 중 오류: {str(e)}")
        except NameError:
            print(f"오류: 설정 파일 저장 중 오류: {str(e)}")

##########################
# 설정 클래스
##########################
class Config:
    def __init__(self):
        # 기본값 설정
        self.min_size = 50
        self.max_size = 300
        self.yolo_confidence_value = 0.35
        self.gravity_direction_threshold = 7
        self.batch_size = 4
        self.output_dir = "output"
        self.distance_trash = 300
        self.min_frame_count_for_violation = 7
        self.horizontal_direction_threshold = 5
        self.vehicle_overlap_threshold = 0.01
        self.max_vehicle_distance = 200
        self.detection_logic = "ALL"
        self.debug_detection = False
        self.performance_monitoring = True
        self.target_fps = 15.0
        self.max_gpu_memory_percent = 80.0
        self.max_cpu_percent = 70.0
        self.performance_warning_threshold = 0.8
        self.performance_log_interval = 5
        
        # 설정 파일에서 값 로드
        self.load_from_file()
    
    def load_from_file(self, file_path="default_settings.txt"):
        """설정 파일에서 값을 로드하여 현재 인스턴스에 적용"""
        settings = load_settings_from_file(file_path)
        
        # 로드된 설정값으로 현재 인스턴스 업데이트
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
                try:
                    logger.debug(f"설정 업데이트: {key} = {value}")
                except NameError:
                    pass  # logger가 없으면 무시
    
    def save_to_file(self, file_path="default_settings.txt"):
        """현재 설정값을 파일에 저장"""
        save_settings_to_file(self, file_path)


# Config 객체 생성
config = Config()


def update_config(new_config):
    """
    외부(예: ui.py)에서 전달받은 설정값을 글로벌 config에 반영하고 파일에 저장.
    """
    global config
    try:
        # 기존 설정값 업데이트
        config.min_size = new_config.min_size
        config.max_size = new_config.max_size
        config.yolo_confidence_value = new_config.yolo_confidence_value
        config.gravity_direction_threshold = new_config.gravity_direction_threshold
        config.distance_trash = new_config.distance_trash
        config.batch_size = new_config.batch_size
        config.detection_logic = new_config.detection_logic
        config.horizontal_direction_threshold = getattr(new_config, 'horizontal_direction_threshold', 5)
        config.max_vehicle_distance = getattr(new_config, 'max_vehicle_distance', 200)
        config.debug_detection = new_config.debug_detection

        # 설정 파일에 저장
        config.save_to_file()

        logger.info("Config 업데이트 및 파일 저장 완료:")
        logger.info(f"min_size={config.min_size}, max_size={config.max_size}, "
                    f"yolo_confidence={config.yolo_confidence_value}, "
                    f"gravity_direction_threshold={config.gravity_direction_threshold}, "
                    f"distance_trash={config.distance_trash}, "
                    f"batch_size={config.batch_size}, "
                    f"detection_logic={config.detection_logic}, "
                    f"horizontal_threshold={config.horizontal_direction_threshold}, "
                    f"max_vehicle_distance={config.max_vehicle_distance}, "
                    f"debug_detection={config.debug_detection}")
    except Exception as e:
        logger.error(f"설정 업데이트 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())


##########################
# LOG 디렉토리 및 로그 파일 설정
##########################
LOG_DIR = "LOG"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"video_processing_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

########################################
# CUDA/GPU 정보 표시 및 로깅
########################################
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

# CUDA 로거 설정 (파일 기록용)
cuda_logger = logging.getLogger("cuda_info")
cuda_logger.setLevel(logging.INFO)
cuda_logger.propagate = False

cuda_file_handler = logging.FileHandler(log_filename)
cuda_file_handler.setLevel(logging.INFO)
cuda_logger.addHandler(cuda_file_handler)

# 로그 파일에도 기록
cuda_logger.info(f"CUDA Available: {cuda_available}")
if cuda_available:
    try:
        cuda_logger.info(f"CUDA Version: {torch.version.cuda}")
        cuda_logger.info(f"PyTorch Version: {torch.__version__}")
        cuda_logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
        cuda_logger.info(f"Device Count: {torch.cuda.device_count()}")
        cuda_logger.info(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    except Exception as e:
        cuda_logger.error(f"CUDA 정보 가져오기 오류: {str(e)}")

cuda_logger.info(f"Using device: {device}")

torch.backends.cudnn.enabled = True

# 출력 디렉토리 생성
os.makedirs(config.output_dir, exist_ok=True)


def exception_hook(exctype, value, tb):
    logger.error("Uncaught exception: " + ''.join(traceback.format_exception(exctype, value, tb)))
    sys.__excepthook__(exctype, value, tb)


sys.excepthook = exception_hook


##########################
# PerformanceMonitor 클래스
##########################
class PerformanceMonitor:
    """실시간 성능 모니터링 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.frame_count = 0
        self.last_log_time = time.time()
        self.fps_history = deque(maxlen=30)  # 최근 30초간 FPS 기록
        
        # 성능 로그 파일 설정
        self.performance_log_file = os.path.join(LOG_DIR, f"performance_benchmark_{datetime.now().strftime('%Y%m%d')}.log")
        
        # 성능 로거 설정
        self.perf_logger = logging.getLogger("performance")
        self.perf_logger.setLevel(logging.INFO)
        self.perf_logger.propagate = False
        
        # 성능 로그 핸들러 (기존 핸들러와 중복 방지)
        if not self.perf_logger.handlers:
            perf_handler = logging.FileHandler(self.performance_log_file)
            perf_handler.setLevel(logging.INFO)
            perf_formatter = logging.Formatter('%(asctime)s - %(message)s')
            perf_handler.setFormatter(perf_formatter)
            self.perf_logger.addHandler(perf_handler)
        
        logger.info(f"PerformanceMonitor 초기화 완료 - 로그 파일: {self.performance_log_file}")
        
        # 초기 벤치마크 정보 기록
        self._log_system_info()
    
    def _log_system_info(self):
        """시스템 정보를 로그에 기록"""
        try:
            import psutil
            
            # CPU 정보
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            
            # GPU 정보
            gpu_info = "N/A"
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_info = f"{gpu_name} ({gpu_memory_gb:.1f}GB)"
            
            system_info = f"""
=== 시스템 성능 벤치마크 시작 ===
CPU: {cpu_count}코어 ({cpu_count_logical}논리코어)
메모리: {total_memory_gb:.1f}GB
GPU: {gpu_info}
목표 성능: FPS≥{self.config.target_fps}, GPU≤{self.config.max_gpu_memory_percent}%, CPU≤{self.config.max_cpu_percent}%
==================================="""
            
            self.perf_logger.info(system_info)
            
        except ImportError:
            self.perf_logger.info("=== 성능 모니터링 시작 (psutil 없음) ===")
        except Exception as e:
            self.perf_logger.error(f"시스템 정보 수집 중 오류: {str(e)}")
    
    def update_frame_count(self):
        """프레임 카운트 업데이트"""
        self.frame_count += 1
        
        # 주기적으로 성능 로그 기록
        current_time = time.time()
        if current_time - self.last_log_time >= self.config.performance_log_interval:
            self._log_performance_metrics()
            self.last_log_time = current_time
    
    def calculate_fps(self):
        """현재 FPS 계산"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            current_fps = self.frame_count / elapsed_time
            self.fps_history.append(current_fps)
            return current_fps
        return 0.0
    
    def get_average_fps(self):
        """평균 FPS 반환"""
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0
    
    def _log_performance_metrics(self):
        """성능 지표를 로그에 기록"""
        try:
            current_fps = self.calculate_fps()
            avg_fps = self.get_average_fps()
            
            # CPU 사용률
            cpu_percent = "N/A"
            memory_percent = "N/A"
            
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
            except ImportError:
                pass
            
            # GPU 메모리 사용률
            gpu_memory_percent = "N/A"
            gpu_memory_used = "N/A"
            gpu_memory_total = "N/A"
            
            if torch.cuda.is_available():
                try:
                    gpu_memory_used_bytes = torch.cuda.memory_allocated(0)
                    gpu_memory_total_bytes = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_used = gpu_memory_used_bytes / (1024**3)  # GB
                    gpu_memory_total = gpu_memory_total_bytes / (1024**3)  # GB
                    gpu_memory_percent = (gpu_memory_used_bytes / gpu_memory_total_bytes) * 100
                except Exception as e:
                    logger.warning(f"GPU 메모리 정보 수집 실패: {str(e)}")
            
            # 성능 지표 로그 기록
            perf_data = f"FPS: {current_fps:.2f} (평균: {avg_fps:.2f}) | " \
                       f"CPU: {cpu_percent}% | " \
                       f"RAM: {memory_percent}% | " \
                       f"GPU메모리: {gpu_memory_percent}%" \
                       f" ({gpu_memory_used}GB/{gpu_memory_total}GB)"
            
            self.perf_logger.info(perf_data)
            
            # 성능 경고 체크
            self._check_performance_warnings(current_fps, cpu_percent, gpu_memory_percent)
            
        except Exception as e:
            self.perf_logger.error(f"성능 지표 수집 중 오류: {str(e)}")
    
    def _check_performance_warnings(self, current_fps, cpu_percent, gpu_memory_percent):
        """성능 경고 확인 및 알림"""
        warnings = []
        
        # FPS 경고
        if isinstance(current_fps, (int, float)) and current_fps < self.config.target_fps * self.config.performance_warning_threshold:
            warnings.append(f"FPS 저하: {current_fps:.2f} < {self.config.target_fps * self.config.performance_warning_threshold:.2f}")
        
        # CPU 경고
        if isinstance(cpu_percent, (int, float)) and cpu_percent > self.config.max_cpu_percent:
            warnings.append(f"CPU 사용률 높음: {cpu_percent:.1f}% > {self.config.max_cpu_percent}%")
        
        # GPU 메모리 경고
        if isinstance(gpu_memory_percent, (int, float)) and gpu_memory_percent > self.config.max_gpu_memory_percent:
            warnings.append(f"GPU 메모리 사용률 높음: {gpu_memory_percent:.1f}% > {self.config.max_gpu_memory_percent}%")
        
        # 경고 로그 출력
        if warnings:
            warning_msg = " | ".join(warnings)
            self.perf_logger.warning(f"⚠️ 성능 경고: {warning_msg}")
            logger.warning(f"성능 경고: {warning_msg}")
    
    def get_performance_summary(self):
        """성능 요약 정보 반환"""
        try:
            current_fps = self.calculate_fps()
            avg_fps = self.get_average_fps()
            
            # 기본 정보
            summary = {
                'current_fps': current_fps,
                'average_fps': avg_fps,
                'target_fps': self.config.target_fps,
                'total_frames': self.frame_count,
                'uptime_seconds': time.time() - self.start_time
            }
            
            # CPU/메모리 정보 추가
            try:
                import psutil
                summary['cpu_percent'] = psutil.cpu_percent()
                summary['memory_percent'] = psutil.virtual_memory().percent
            except ImportError:
                summary['cpu_percent'] = None
                summary['memory_percent'] = None
            
            # GPU 정보 추가
            if torch.cuda.is_available():
                try:
                    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    summary['gpu_memory_used_gb'] = gpu_memory_used
                    summary['gpu_memory_total_gb'] = gpu_memory_total
                    summary['gpu_memory_percent'] = (gpu_memory_used / gpu_memory_total) * 100
                except:
                    summary['gpu_memory_percent'] = None
            else:
                summary['gpu_memory_percent'] = None
            
            return summary
            
        except Exception as e:
            logger.error(f"성능 요약 생성 중 오류: {str(e)}")
            return {'error': str(e)}
    
    def reset_counters(self):
        """카운터 리셋"""
        self.start_time = time.time()
        self.frame_count = 0
        self.fps_history.clear()
        self.perf_logger.info("성능 카운터 리셋")


##########################
# EuclideanDistTracker 클래스
##########################
class EuclideanDistTracker:
    """간단한 유클리드 거리 기반 객체 추적기"""

    def __init__(self):
        self.objects = {}  # {object_id: (x, y, w, h)}
        self.next_object_id = 0
        self.distance_threshold = 50

    def update(self, detections):
        if not detections:
            return []

        updated_objects = {}
        assigned_ids = set()

        for (new_x, new_y, new_w, new_h) in detections:
            matched = False
            new_center = (new_x + new_w // 2, new_y + new_h // 2)
            for object_id, (old_x, old_y, old_w, old_h) in self.objects.items():
                if object_id in assigned_ids:
                    continue
                old_center = (old_x + old_w // 2, old_y + old_h // 2)
                distance = math.sqrt((new_center[0] - old_center[0]) ** 2 + (new_center[1] - old_center[1]) ** 2)

                if distance < self.distance_threshold:
                    updated_objects[object_id] = (new_x, new_y, new_w, new_h)
                    assigned_ids.add(object_id)
                    matched = True
                    break

            if not matched:
                updated_objects[self.next_object_id] = (new_x, new_y, new_w, new_h)
                assigned_ids.add(self.next_object_id)
                self.next_object_id += 1

        self.objects = updated_objects
        return [(x, y, w, h, object_id) for object_id, (x, y, w, h) in updated_objects.items()]


##########################
# ROI 설정 저장/로드 함수
##########################
ROI_SETTINGS_FILE = "roi_settings.json"


def save_roi_settings(roi_x, roi_y, roi_width, roi_height, min_size, max_size):
    """ROI 설정을 JSON 파일로 저장"""
    try:
        roi_data = {
            "roi_x": roi_x,
            "roi_y": roi_y,
            "roi_width": roi_width,
            "roi_height": roi_height,
            "min_size": min_size,
            "max_size": max_size,
        }
        with open(ROI_SETTINGS_FILE, "w") as file:
            json.dump(roi_data, file, indent=4)
        logger.info("ROI 설정 저장 완료")
    except Exception as e:
        logger.error(f"ROI 설정 저장 실패: {str(e)}")


def load_roi_settings():
    """JSON 파일에서 ROI 설정 로드, 실패 시 기본값 반환"""
    logger.debug("load_roi_settings 함수 호출됨.")
    try:
        with open(ROI_SETTINGS_FILE, "r") as file:
            roi_data = json.load(file)
            logger.info(f"ROI 데이터 로드 성공: {roi_data}")
            return (
                max(0, roi_data.get("roi_x", 0)),
                max(0, roi_data.get("roi_y", 300)),
                max(1, roi_data.get("roi_width", 800)),
                max(1, roi_data.get("roi_height", 150)),
                max(1, roi_data.get("min_size", 50)),
                max(1, roi_data.get("max_size", 400)),
            )
    except FileNotFoundError:
        logger.warning(f"ROI 설정 파일이 존재하지 않습니다. 기본값을 사용합니다.")
        return 0, 300, 800, 150, 50, 400
    except json.JSONDecodeError as e:
        logger.error(f"ROI 설정 파일 형식이 잘못되었습니다: {str(e)}")
        return 0, 300, 800, 150, 50, 400
    except Exception as e:
        logger.error(f"ROI 데이터 로드 실패: {str(e)}, 기본값 사용")
        return 0, 300, 800, 150, 50, 400


##########################
# VideoWriterThread 클래스
##########################
class VideoWriterThread(threading.Thread):
    """비디오 저장을 담당하는 스레드 클래스"""

    def __init__(self, write_queue, config, roi_x, roi_y, roi_width, roi_height):
        super().__init__()
        self.write_queue = write_queue
        self.config = config
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.daemon = True
        self._stop_event = threading.Event()
        # 수정: video_saved_signal을 직접 전달받지는 않음 (메인 VideoThread에서 관리)
        self.start()

    def stop(self):
        """스레드 정지 신호를 설정"""
        self._stop_event.set()
        # 종료 신호를 큐에 넣어 블로킹 상태의 get()를 해제
        self.write_queue.put((None, None))

    def stopped(self):
        """스레드가 정지 상태인지 확인"""
        return self._stop_event.is_set()

    def run(self):
        """메인 실행 루프"""
        while not self.stopped():
            try:
                # 타임아웃을 설정하여 주기적으로 정지 신호를 확인
                pre_frames, roi_frames = self.write_queue.get(timeout=1.0)
                if pre_frames is None and roi_frames is None:
                    if self.stopped():
                        break
                    continue

                video_path = self.capture_video(pre_frames, roi_frames)
                self.write_queue.task_done()

                # 비디오 저장 완료 시 경로 기록 (로그만 남김, 신호는 메인 스레드에서 처리)
                if video_path:
                    logger.info(f"VideoWriterThread: 비디오 저장 완료 - {video_path}")
            except queue.Empty:
                # 타임아웃 발생 - 정지 신호 확인 후 계속
                continue
            except Exception as e:
                logger.error(f"VideoWriterThread 오류: {str(e)}")
                logger.error(traceback.format_exc())

    def capture_video(self, pre_frames, roi_frames=None):
        """이벤트 감지 시 비디오 저장"""
        if not pre_frames or len(pre_frames) == 0:
            logger.error("저장할 프레임이 없습니다.")
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.config.output_dir, f"trash_event_{timestamp}.avi")

        fps = 30  # 더 부드러운 영상을 위해 FPS 증가

        # 첫 프레임을 사용하여 비디오 크기 결정
        sample_frame = pre_frames[0]
        frame_height, frame_width = sample_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        try:
            # 전체 프레임 크기로 저장
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            logger.info(
                f"쓰레기 투기 감지! 영상 저장 시작: {video_path} (크기: {frame_width}x{frame_height}, FPS: {fps}, 프레임 수: {len(pre_frames)})")

            # 모든 프레임을 순서대로 저장
            for frame in pre_frames:
                out.write(frame)

            # roi_frames가 있으면 추가로 저장 (현재 코드에서는 사용하지 않음)
            if roi_frames and len(roi_frames) > 0:
                for frame in roi_frames:
                    out.write(frame)

            out.release()
            logger.info(f"영상 저장 완료: {video_path} (총 {len(pre_frames)} 프레임)")

            # 비디오 저장 완료 후 경로 반환 - 이 비디오 경로가 VideoThread로 전달됨
            return video_path

        except Exception as e:
            logger.error(f"영상 저장 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            if 'out' in locals():
                out.release()
            return None


##########################
# FrameReaderThread 클래스
##########################
class FrameReaderThread(threading.Thread):
    """
    OpenCV VideoCapture 디코딩을 별도 스레드에서 수행해,
    I/O 병목을 줄이기 위한 클래스.
    """

    def __init__(self, video_path, frame_queue, run_flag):
        super().__init__()
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.run_flag = run_flag  # VideoThread와 공유(멈춤 시점을 체크)
        self.daemon = True
        self.start()

    def run(self):
        """프레임 읽기 메인 루프"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logger.error(f"디코딩 스레드: 비디오를 열 수 없습니다: {self.video_path}")
                self.frame_queue.put(None)  # 종료 신호
                return

            frame_count = 0
            while self.run_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"디코딩 스레드: 더 이상 프레임이 없음(EOF), 총 {frame_count} 프레임 처리됨")
                    break

                # 큐가 가득 차면 잠시 대기 (I/O 폭주 방지)
                while self.run_flag.is_set() and self.frame_queue.full():
                    time.sleep(0.01)

                # 프레임을 큐에 넣는다
                if self.run_flag.is_set():  # 중간에 정지 신호가 발생할 수 있으므로 다시 확인
                    self.frame_queue.put(frame)
                    frame_count += 1

            cap.release()
            # 종료 신호로 None을 푸시
            self.frame_queue.put(None)
            logger.info("디코딩 스레드 종료")
        except Exception as e:
            logger.error(f"디코딩 스레드에서 예외 발생: {str(e)}")
            logger.error(traceback.format_exc())
            self.frame_queue.put(None)


##########################
# VideoThread 클래스
##########################
class VideoThread(QThread):
    """비디오 처리를 담당하는 QThread 클래스"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    video_saved_signal = pyqtSignal(str)  # 비디오 저장 완료 시그널 추가
    littering_detected_signal = pyqtSignal(tuple, str, float, dict)  # 쓰레기 투기 감지 시그널 추가

    def __init__(
            self, video_path, min_size, max_size, min_box_size,
            roi_x, roi_y, roi_width, roi_height, config
    ):
        super().__init__()
        self.video_path = video_path
        self.min_size = min_size
        self.max_size = max_size
        self.min_box_size = min_box_size
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        self._run_flag = True
        self.config = config

        # 배치 처리 관련 락
        self.batch_lock = threading.Lock()
        
        # 디바이스 설정 저장
        self.device = device

        try:
            # 싱글톤 패턴으로 모델 로드
            self.model = load_model_with_spinner("yolov8n.pt")

            # GPU/CUDA 상태 표시
            print("\n" + "🔥"*30)
            print("🎯 YOLO 모델 GPU 최적화 시작")
            print("🔥"*30)

            # 안전한 모델 초기화 (CUDA 호환성 문제 해결)
            try:
                # CUDA가 사용 가능하고 안전한 경우에만 GPU 사용
                if torch.cuda.is_available():
                    print("⚡ GPU 사용 가능 - GPU로 모델 이동 중...")
                    # GPU로 모델 이동
                    self.model.to(self.device)
                    
                    # CUDA 버전 호환성 확인 후 최적화 적용
                    try:
                        print("🔧 모델 최적화 중 (fuse + half precision)...")
                        # float 상태에서 fuse 시도
                        self.model.float()
                        self.model.fuse()
                        # fuse 성공 시 half precision 적용
                        if self.device.type == 'cuda':
                            self.model.half()
                        print("✅ GPU 최적화 완료! (fuse + half precision)")
                        logger.info("모델 GPU 최적화 완료 (fuse + half precision)")
                    except RuntimeError as cuda_err:
                        # CUDA 오류 발생 시 기본 float로 폴백
                        print(f"⚠️  CUDA 최적화 실패: {str(cuda_err)}")
                        print("🔄 CPU 모드로 폴백 중...")
                        logger.warning(f"CUDA 최적화 실패, CPU로 폴백: {str(cuda_err)}")
                        self.model.cpu().float()
                        self.device = torch.device("cpu")
                        print("💻 CPU 모드로 전환 완료")
                        logger.info("CPU 모드로 전환됨")
                else:
                    # CUDA가 없으면 CPU 사용
                    print("💻 CUDA 없음 - CPU 모드로 초기화 중...")
                    self.model.cpu().float()
                    self.device = torch.device("cpu")
                    print("✅ CPU 모드 초기화 완료")
                    logger.info("CPU 모드로 모델 초기화")
                    
            except Exception as model_opt_err:
                # 모델 최적화 실패 시 기본 설정으로 폴백
                print(f"❌ 모델 최적화 오류: {str(model_opt_err)}")
                print("🔄 기본 CPU 모드로 폴백 중...")
                logger.error(f"모델 최적화 중 오류: {str(model_opt_err)}")
                self.model.cpu().float()
                self.device = torch.device("cpu")
                print("✅ 기본 CPU 모드 폴백 완료")
                logger.info("기본 CPU 모드로 폴백 완료")

            print(f"🎯 최종 사용 장치: {self.device}")
            print("🔥"*30 + "\n")
            logger.info("모델 로드 및 초기화 성공")
        except Exception as e:
            logger.error(f"모델 초기화 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # 추적기 및 상태 변수
        self.tracker = EuclideanDistTracker()
        self.object_movements = {}
        self.gravity_direction_threshold = 6
        self.detected_vehicles = set()
        self.vehicle_tracking = {}
        self.violation_vehicles = set()
        self.vehicle_classes = [2, 5, 7]  # car, bus, truck
        
        # DetectionStrategyManager 초기화 및 전략 등록
        self.strategy_manager = DetectionStrategyManager()
        self._initialize_detection_strategies()

        # 프레임 버퍼 설정
        self.continuous_buffer = deque(maxlen=300)  # 적절한 크기로 초기화
        self.continuous_buffer_maxlen = 300  # 나중에 fps에 맞게 조정됨
        self.buffer_lock = threading.Lock()
        self.buffer_duration = 2  # 이벤트 전 버퍼링할 시간(초)
        self.post_event_duration = 5  # 이벤트 후 녹화할 시간(초)
        self.fps = 30  # 기본값, 실제 값은 validate_video_file에서 설정됨
        self.current_frame_index = 0

        # VideoWriterThread 초기화 전 필요한 큐
        self.write_queue = queue.Queue()
        self.writer_thread = None

        # 이벤트 관련 변수
        self.event_lock = threading.Lock()
        self.event_active = False
        self.event_triggered_at = 0

        # 영상 디코딩용
        self.frame_queue = queue.Queue(maxsize=20)  # 적절한 큐 크기 설정
        self.run_flag = threading.Event()
        self.run_flag.set()  # True 상태
        
        # 성능 모니터링 초기화
        if self.config.performance_monitoring:
            self.performance_monitor = PerformanceMonitor(self.config)
            logger.info("성능 모니터링 활성화")
        else:
            self.performance_monitor = None
            
        # 캐시 관리자 초기화 (고성능 버전 사용)
        self.cache_manager = create_high_performance_cache(self.config)
        logger.info("고성능 캐시 관리자 초기화 완료")

    def _initialize_detection_strategies(self):
        """감지 전략들을 등록하고 활성화하는 함수"""
        try:
            # 전략 등록
            self.strategy_manager.register_strategy("size_range", SizeRangeStrategy())
            self.strategy_manager.register_strategy("gravity_direction", GravityDirectionStrategy())
            self.strategy_manager.register_strategy("vehicle_distance", VehicleDistanceStrategy())
            self.strategy_manager.register_strategy("vehicle_overlap", VehicleOverlapStrategy())
            self.strategy_manager.register_strategy("direction_alignment", DirectionAlignmentStrategy())
            self.strategy_manager.register_strategy("vehicle_association", VehicleAssociationStrategy())
            
            # 기본 전략 활성화
            self.strategy_manager.enable_strategy("size_range")
            self.strategy_manager.enable_strategy("gravity_direction")
            self.strategy_manager.enable_strategy("vehicle_distance")
            self.strategy_manager.enable_strategy("direction_alignment")  # DirectionAlignmentStrategy 활성화 추가
            
            # 활성화된 전략 목록 로깅
            enabled_strategies = self.strategy_manager.get_enabled_strategies()
            strategy_names = [strategy.name() for strategy in enabled_strategies.values()]
            logger.info(f"DetectionStrategyManager 초기화 완료: {len(enabled_strategies)}개 전략 활성화")
            logger.info(f"활성화된 전략들: {', '.join(strategy_names)}")
            
        except Exception as e:
            logger.error(f"전략 초기화 중 오류: {str(e)}")
            logger.error(traceback.format_exc())

    def debug_detection_info(self, obj_id, bbox, is_detected, strategies_result=None):
        """객체 검출 정보를 디버깅 출력하는 함수 - 향상된 버전"""
        if not self.config.debug_detection:
            return  # 디버깅 모드가 아니면 출력하지 않음

        # bbox 형식 정규화 및 좌표 추출
        try:
            # bbox가 tuple 안에 tuple인 경우 처리
            if isinstance(bbox, tuple) and len(bbox) == 2 and isinstance(bbox[0], tuple):
                bbox = bbox[0]  # 첫 번째 tuple만 사용
            
            if len(bbox) == 4:
                x, y, w, h = bbox
                # w, h가 음수인 경우 (x1,y1,x2,y2) 형식으로 간주
                if w < 0 or h < 0:
                    x1, y1, x2, y2 = bbox
                    x, y = x1, y1
                    w, h = x2 - x1, y2 - y1
                    # 여전히 음수면 절댓값 사용
                    w, h = abs(w), abs(h)
            else:
                x, y, w, h = 0, 0, 0, 0
                print(f"⚠️  경고: 잘못된 bbox 형식: {bbox}")
                
        except Exception as e:
            x, y, w, h = 0, 0, 0, 0
            print(f"⚠️  경고: bbox 처리 중 오류: {e}, bbox: {bbox}")
        
        status = "✅성공" if is_detected else "❌실패"
        area = w * h
        
        # 객체 추적 정보 추가
        tracking_info = self.object_movements.get(obj_id, {})
        frame_count = tracking_info.get("count", 0)
        trajectory_length = len(tracking_info.get("trajectory", []))
        
        print(f"\n{'='*60}")
        print(f"🎯 [객체 추적 디버깅] ID: {obj_id}")
        print(f"📍 좌표: ({x}, {y}) 크기: {w} x {h} (면적: {area} 픽셀)")
        print(f"📊 프레임 카운트: {frame_count}/{self.config.min_frame_count_for_violation}")
        print(f"📈 궤적 길이: {trajectory_length}")
        print(f"🎯 최종 검출: {status}")
        print(f"⚙️  로직 모드: {self.config.detection_logic}")
        
        # 크기 범위 체크 추가
        size_status = "✅" if self.config.min_size <= area <= self.config.max_size else "❌"
        print(f"📏 크기 체크: {size_status} (범위: {self.config.min_size} ~ {self.config.max_size} 픽셀)")

        # 전략별 결과가 있다면 상세 출력
        if strategies_result and isinstance(strategies_result, dict):
            print(f"\n📋 [전략별 상세 결과]")
            for strategy_id, result in strategies_result.items():
                status_icon = "✅" if result else "❌"
                strategy_name = "알 수 없는 전략"
                
                # 전략 이름 가져오기
                if hasattr(self, 'strategy_manager') and strategy_id in self.strategy_manager.strategies:
                    strategy_name = self.strategy_manager.strategies[strategy_id].name()
                
                print(f"  {status_icon} {strategy_id} ({strategy_name}): {result}")

        # 객체 궤적 정보 출력
        if trajectory_length > 1:
            trajectory = tracking_info.get("trajectory", [])
            recent_positions = trajectory[-min(5, len(trajectory)):]  # 최근 5개 위치
            
            print(f"\n📍 [최근 이동 경로] (최근 {len(recent_positions)}개 위치)")
            for i, pos_info in enumerate(recent_positions):
                center = pos_info.get('center', (0, 0))
                print(f"  {i+1}. ({center[0]}, {center[1]})")
                
            # 이동 방향 계산 및 출력
            if len(recent_positions) >= 2:
                first_pos = recent_positions[0]['center']
                last_pos = recent_positions[-1]['center']
                dx = last_pos[0] - first_pos[0]
                dy = last_pos[1] - first_pos[1]
                
                direction = "정지"
                if abs(dx) > 3 or abs(dy) > 3:  # 최소 이동 거리
                    if abs(dx) > abs(dy):
                        direction = "우측" if dx > 0 else "좌측"
                    else:
                        direction = "하강" if dy > 0 else "상승"
                
                print(f"  🧭 이동 방향: {direction} (dx: {dx:+.1f}, dy: {dy:+.1f})")

        print(f"{'='*60}\n")

    def debug_strategy_details(self, obj_id, tracking_info, vehicle_info):
        """전략별 상세 분석 정보를 출력하는 함수"""
        if not self.config.debug_detection:
            return

        if not tracking_info or len(tracking_info) < 1:
            return

        print(f"\n🔍 [전략별 상세 분석] 객체 ID: {obj_id}")
        
        # 현재 객체 정보
        current_info = tracking_info[-1]
        current_center = current_info['center']
        current_bbox = current_info['bbox']
        
        # 1. 크기 범위 전략 분석
        width = current_bbox[2] - current_bbox[0]
        height = current_bbox[3] - current_bbox[1]
        area = width * height
        size_valid = self.config.min_size <= area <= self.config.max_size
        print(f"📏 크기 분석: 면적={area} (범위: {self.config.min_size}-{self.config.max_size}) {'✅' if size_valid else '❌'}")
        
        # 2. 중력 방향 전략 분석
        if len(tracking_info) >= 2:
            y_movements = []
            for i in range(1, len(tracking_info)):
                prev_y = tracking_info[i-1]['center'][1]
                curr_y = tracking_info[i]['center'][1]
                y_diff = curr_y - prev_y
                y_movements.append(y_diff)
            
            downward_moves = sum(1 for diff in y_movements if diff >= 1)
            total_moves = len(y_movements)
            downward_ratio = downward_moves / total_moves if total_moves > 0 else 0
            
            print(f"⬇️  중력 분석: 하강 움직임 {downward_moves}/{total_moves} ({downward_ratio:.1%}) {'✅' if downward_ratio >= 0.8 else '❌'}")
        
        # 3. 차량 거리 전략 분석
        if vehicle_info:
            min_distance = float('inf')
            closest_vehicle = None
            
            for i, vehicle in enumerate(vehicle_info):
                veh_center = vehicle['center']
                
                # 캐시를 사용한 거리 계산
                if hasattr(self, 'cache_manager'):
                    distance = self.cache_manager.get_optimized_distance(current_center, veh_center)
                else:
                    distance = math.sqrt((current_center[0] - veh_center[0])**2 + (current_center[1] - veh_center[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle = i
            
            distance_valid = min_distance <= self.config.distance_trash
            print(f"🚗 차량 거리: 최근접={min_distance:.1f}px (기준: {self.config.distance_trash}px) {'✅' if distance_valid else '❌'}")
            
            if closest_vehicle is not None:
                vehicle = vehicle_info[closest_vehicle]
                veh_bbox = vehicle['bbox']
                print(f"    가장 가까운 차량: ({veh_bbox[0]}, {veh_bbox[1]}, {veh_bbox[2]}, {veh_bbox[3]})")
        else:
            print(f"🚗 차량 거리: 차량 정보 없음 ❌")
        
        # 4. 차량 겹침 전략 분석
        if vehicle_info:
            overlap_detected = False
            for vehicle in vehicle_info:
                veh_bbox = vehicle['bbox']
                
                # 중심점이 차량 내부에 있는지 확인
                if (veh_bbox[0] <= current_center[0] <= veh_bbox[2] and
                    veh_bbox[1] <= current_center[1] <= veh_bbox[3]):
                    overlap_detected = True
                    print(f"🚫 차량 겹침: 객체 중심점이 차량 내부에 위치 ❌")
                    break
            
            if not overlap_detected:
                print(f"🚫 차량 겹침: 차량과 겹치지 않음 ✅")
        else:
            print(f"🚫 차량 겹침: 차량 정보 없음 ✅")
        
        print()  # 빈 줄 추가

    def validate_video_file(self, file_path: str) -> bool:
        """
        이 함수에서는 비디오 파일의 기본 유효성만 확인.
        디코딩 스레드는 이후 별도 Thread(FrameReaderThread)에서 담당.
        """
        try:
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            file_ext = os.path.splitext(file_path)[1].lower()

            if not any(file_ext == ext for ext in valid_extensions):
                logger.warning(f"지원되지 않는 파일 형식: {file_path} (확장자: {file_ext})")
                return False

            if not os.path.exists(file_path):
                logger.error(f"파일을 찾을 수 없음: {file_path}")
                return False

            # 메타데이터만 일단 확인
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"비디오 파일을 열 수 없음: {file_path}")
                return False

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if total_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
                logger.warning(f"잘못된 비디오 메타데이터: {file_path} (프레임 수: {total_frames}, FPS: {fps}, 해상도: {width}x{height})")
                return False

            self.fps = fps
            self.config.fps = self.fps

            # 버퍼 크기 설정 - 이벤트 전후 시간을 고려하여 계산
            total_buffer_duration = self.buffer_duration + self.post_event_duration + 1
            self.continuous_buffer_maxlen = int(self.fps * total_buffer_duration)
            self.continuous_buffer = deque(maxlen=self.continuous_buffer_maxlen)

            logger.info(
                f"FPS 설정: {self.fps}, 연속 버퍼 크기: {self.continuous_buffer_maxlen} 프레임 (총 {total_buffer_duration}초)")

            # VideoWriterThread 생성
            if self.writer_thread is None:
                self.writer_thread = VideoWriterThread(
                    self.write_queue, self.config, self.roi_x, self.roi_y,
                    self.roi_width, self.roi_height
                )
            return True
        except cv2.error as cv_err:
            logger.error(f"OpenCV 오류: {str(cv_err)}")
            logger.error(traceback.format_exc())
            return False
        except IOError as io_err:
            logger.error(f"파일 IO 오류: {str(io_err)}")
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"비디오 파일 검증 중 예외 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def clear_buffers(self):
        """메모리 정리를 위해 모든 버퍼를 비우는 메서드"""
        with self.buffer_lock:
            self.continuous_buffer.clear()

        with self.batch_lock:
            if hasattr(self, 'batch_frames'):
                self.batch_frames = []
            if hasattr(self, 'batch_originals'):
                self.batch_originals = []

        # 추적 데이터 정리
        self.object_movements.clear()
        self.detected_vehicles.clear()
        self.vehicle_tracking.clear()
        self.violation_vehicles.clear()

        # 큐 비우기
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        # 캐시 정리
        if hasattr(self, 'cache_manager'):
            self.cache_manager.clear_all_caches()
            logger.debug("캐시 관리자 정리 완료")

        logger.debug("모든 버퍼와 추적 데이터 정리 완료")

    def run(self):
        """
        메인 추론 루프:
         1) FrameReaderThread에서 디코딩된 frame을 queue에서 꺼냄
         2) 배치 구성 -> YOLO 추론 -> 후처리
        """
        if not self.validate_video_file(self.video_path):
            logger.error(f"비디오 파일이 유효하지 않습니다: {self.video_path}")
            return

        # 디코딩 스레드 시작
        self.reader_thread = FrameReaderThread(
            self.video_path,
            frame_queue=self.frame_queue,
            run_flag=self.run_flag
        )

        try:
            object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=95)

            frame_count = 0
            start_time = time.time()

            BATCH_SIZE = self.config.batch_size
            batch_frames = []
            batch_originals = []

            while self._run_flag:
                try:
                    # 디코딩 스레드가 큐에 넣은 프레임을 가져옴 (타임아웃 설정)
                    frame = self.frame_queue.get(timeout=1.0)

                    # None이면 EOF 또는 오류 -> 종료
                    if frame is None:
                        logger.info("VideoThread: 프레임이 None -> 디코딩 종료 신호")
                        # 이벤트가 활성화된 상태에서 EOF 도달 시 강제 저장
                        if self.event_active:
                            self.collect_post_event_frame(None, roi_x1, roi_y1)
                        break

                    self.current_frame_index += 1
                    process_frame = frame.copy()

                    # 연속 버퍼에 프레임 추가 (이벤트 전후 녹화용)
                    with self.buffer_lock:
                        self.continuous_buffer.append(frame.copy())

                    # ROI 영역이 프레임 경계를 벗어나지 않도록 보정
                    roi_x1 = max(0, min(self.roi_x, frame.shape[1]))
                    roi_y1 = max(0, min(self.roi_y, frame.shape[0]))
                    roi_x2 = max(0, min(self.roi_x + self.roi_width, frame.shape[1]))
                    roi_y2 = max(0, min(self.roi_y + self.roi_height, frame.shape[0]))

                    # ROI 영역 표시
                    cv2.rectangle(process_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

                    # ROI 추출
                    if roi_y2 > roi_y1 and roi_x2 > roi_x1:  # 유효한 ROI 크기 확인
                        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                    else:
                        logger.warning(
                            f"유효하지 않은 ROI 크기: x={roi_x1}, y={roi_y1}, width={roi_x2 - roi_x1}, height={roi_y2 - roi_y1}")
                        continue

                    # 배치 쌓기 (락 사용)
                    with self.batch_lock:
                        batch_frames.append(roi_frame)
                        batch_originals.append(process_frame)

                    # 배치 추론
                    if len(batch_frames) >= BATCH_SIZE:
                        with self.batch_lock:
                            # 현재 배치 복사 후 배치 초기화
                            current_batch_frames = batch_frames.copy()
                            current_batch_originals = batch_originals.copy()
                            batch_frames.clear()
                            batch_originals.clear()

                        # 배치 추론 실행
                        try:
                            results = self.model(
                                current_batch_frames,
                                verbose=False,
                                device=self.device
                            )

                            for i, result in enumerate(results):
                                original_frame = current_batch_originals[i]
                                roi_for_bg = current_batch_frames[i]

                                yolo_boxes = self.parse_yolo_results(result, roi_x1, roi_y1)

                                contour_detections = self.detect_litter_candidates(
                                    roi_for_bg, object_detector, roi_x1, roi_y1, yolo_boxes
                                )

                                tracked_objects = self.tracker.update(contour_detections)
                                self.analyze_tracked_objects(
                                    original_frame, tracked_objects, yolo_boxes
                                )
                                self.mark_vehicles(original_frame, yolo_boxes)

                                self.change_pixmap_signal.emit(original_frame)

                                # 이벤트가 활성화된 경우 collect_post_event_frame 호출
                                if self.event_active:
                                    self.collect_post_event_frame(frame, roi_x1, roi_y1)

                                # 성능 모니터링 업데이트
                                if self.performance_monitor:
                                    self.performance_monitor.update_frame_count()
                                
                                # FPS 계산 및 출력 (기존 로직 유지하되 성능 모니터와 통합)
                                frame_count += 1
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 1.0:
                                    fps_show = frame_count / elapsed_time
                                    if not self.performance_monitor:  # 성능 모니터가 없을 때만 기존 로그 출력
                                        logger.info(f"현재 FPS: {fps_show:.2f}")
                                    
                                    # 10초마다 캐시 성능 통계 로그 출력
                                    if hasattr(self, 'last_cache_log_time'):
                                        if time.time() - self.last_cache_log_time >= 10.0:
                                            self.log_cache_performance()
                                            self.last_cache_log_time = time.time()
                                    else:
                                        self.last_cache_log_time = time.time()
                                    
                                    frame_count = 0
                                    start_time = time.time()
                        except torch.cuda.OutOfMemoryError:
                            logger.error("CUDA 메모리 부족 오류 발생. 배치 크기를 줄여보세요.")
                        except Exception as e:
                            logger.error(f"배치 처리 중 오류 발생: {str(e)}")
                            logger.error(traceback.format_exc())

                except queue.Empty:
                    # 타임아웃 발생 - 정지 신호 확인
                    if not self._run_flag:
                        break
                    continue

            # 남은 배치 처리
            with self.batch_lock:
                remaining_batch_frames = batch_frames.copy()
                remaining_batch_originals = batch_originals.copy()
                batch_frames.clear()
                batch_originals.clear()

            if remaining_batch_frames:
                try:
                    results = self.model(
                        remaining_batch_frames,
                        verbose=False,
                        device=self.device
                    )
                    for i, result in enumerate(results):
                        original_frame = remaining_batch_originals[i]
                        roi_for_bg = remaining_batch_frames[i]

                        yolo_boxes = self.parse_yolo_results(result, roi_x1, roi_y1)
                        contour_detections = self.detect_litter_candidates(
                            roi_for_bg, object_detector, roi_x1, roi_y1, yolo_boxes
                        )
                        tracked_objects = self.tracker.update(contour_detections)
                        self.analyze_tracked_objects(
                            original_frame, tracked_objects, yolo_boxes
                        )
                        self.mark_vehicles(original_frame, yolo_boxes)
                        self.change_pixmap_signal.emit(original_frame)

                        # 이벤트가 활성화된 경우 collect_post_event_frame 호출
                        if self.event_active:
                            self.collect_post_event_frame(frame, roi_x1, roi_y1)
                        
                        # 성능 모니터링 업데이트 (남은 배치 처리)
                        if self.performance_monitor:
                            self.performance_monitor.update_frame_count()
                except Exception as e:
                    logger.error(f"남은 배치 처리 중 오류: {str(e)}")
                    logger.error(traceback.format_exc())

        except torch.cuda.OutOfMemoryError as cuda_err:
            logger.error(f"CUDA 메모리 부족: {str(cuda_err)}")
            logger.error(traceback.format_exc())
        except cv2.error as cv_err:
            logger.error(f"OpenCV 오류: {str(cv_err)}")
            logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"비디오 처리 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # 스레드 종료 처리
            self.run_flag.clear()

            try:
                if hasattr(self, 'reader_thread') and self.reader_thread is not None:
                    self.reader_thread.join(timeout=2.0)
                    if self.reader_thread.is_alive():
                        logger.warning("FrameReaderThread가 시간 내에 종료되지 않았습니다.")
            except Exception as e:
                logger.error(f"FrameReaderThread 종료 중 오류: {str(e)}")

            try:
                if self.writer_thread:
                    self.writer_thread.stop()
                    self.writer_thread.join(timeout=2.0)
                    if self.writer_thread.is_alive():
                        logger.warning("VideoWriterThread가 시간 내에 종료되지 않았습니다.")
            except Exception as e:
                logger.error(f"VideoWriterThread 종료 중 오류: {str(e)}")

            # 메모리 해제 및 리소스 정리
            self.clear_buffers()
            cv2.destroyAllWindows()
            logger.info("VideoThread: 리소스 정리 완료")

    def get_cache_performance_stats(self):
        """캐시 성능 통계를 반환하는 메서드"""
        if hasattr(self, 'cache_manager'):
            return self.cache_manager.get_performance_stats()
        return None
    
    def log_cache_performance(self):
        """캐시 성능 통계를 로그에 기록 (고성능 버전)"""
        if hasattr(self, 'cache_manager'):
            try:
                cache_stats = self.cache_manager.get_performance_stats()
                
                logger.info("=== 고성능 캐시 성능 통계 ===")
                logger.info(f"전체 캐시 적중률: {cache_stats['overall_hit_rate']:.1f}%")
                logger.info(f"총 요청 수: {cache_stats['total_requests']}")
                logger.info(f"거리 계산 수: {cache_stats['distance_calculations']}")
                logger.info(f"차량 쿼리 수: {cache_stats['vehicle_queries']}")
                
                # 성능 요약
                performance = cache_stats.get('performance_summary', {})
                logger.info(f"캐시 효율성: {performance.get('cache_efficiency', 0):.1f}%")
                logger.info(f"메모리 사용률: {performance.get('memory_usage', 0):.1f}%")
                logger.info(f"평균 접근 횟수: {performance.get('avg_access_per_entry', 0):.1f}")
                
                # 거리 캐시 상세 통계
                distance_cache = cache_stats.get('distance_cache', {})
                logger.info(f"거리 캐시 엔트리: {distance_cache.get('entries', 0)}")
                logger.info(f"차량 캐시 프레임: {cache_stats.get('vehicle_cache_frames', 0)}")
                
                logger.info("========================")
                
            except Exception as e:
                logger.error(f"캐시 성능 통계 로그 중 오류: {str(e)}")

    def analyze_tracked_objects(self, frame, tracked_objects, yolo_boxes):
        roi_x1 = self.roi_x
        roi_y1 = self.roi_y
        roi_x2 = self.roi_x + self.roi_width
        roi_y2 = self.roi_y + self.roi_height

        # 차량 정보 변환 (yolo_boxes를 전략 클래스가 기대하는 형식으로)
        vehicle_info = self.get_vehicle_info_from_boxes(yolo_boxes)

        for (x, y, w, h, obj_id) in tracked_objects:
            current_center = (x + w // 2, y + h // 2)
            area = w * h

            if not (self.min_size <= area <= self.max_size):
                if obj_id in self.object_movements:
                    del self.object_movements[obj_id]
                continue

            cv2.putText(frame, f"ID {obj_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 객체 이력 업데이트
            if obj_id not in self.object_movements:
                # 새 객체 초기화
                self.object_movements[obj_id] = {
                    "trajectory": [{'center': current_center, 'bbox': (x, y, x + w, y + h), 'class_name': 'litter',
                                    'confidence': 1.0}],
                    "count": 1,
                    "video_saved": False,
                    "related_vehicle": None,
                    "movement_direction": None,
                    "last_update": time.time()
                }
            else:
                # 기존 객체의 경우 데이터 형식 확인 및 업데이트
                if not isinstance(self.object_movements[obj_id], dict):
                    # 형식이 잘못된 경우 초기화
                    self.object_movements[obj_id] = {
                        "trajectory": [{'center': current_center, 'bbox': (x, y, x + w, y + h), 'class_name': 'litter',
                                        'confidence': 1.0}],
                        "count": 1,
                        "video_saved": False,
                        "related_vehicle": None,
                        "movement_direction": None,
                        "last_update": time.time()
                    }
                else:
                    # trajectory 키가 없으면 초기화
                    if "trajectory" not in self.object_movements[obj_id]:
                        self.object_movements[obj_id]["trajectory"] = []

                    # 이전 위치와 현재 위치에 기반한 이동 방향 및 거리 계산
                    if len(self.object_movements[obj_id]["trajectory"]) > 0:
                        prev_info = self.object_movements[obj_id]["trajectory"][-1]
                        prev_center = prev_info['center']
                        x_diff = current_center[0] - prev_center[0]
                        y_diff = current_center[1] - prev_center[1]

                        # 이동 방향 계산
                        if abs(x_diff) > abs(y_diff):
                            # 수평 이동이 더 큰 경우
                            movement_direction = "right" if x_diff > 0 else "left"
                        else:
                            # 수직 이동이 더 큰 경우
                            movement_direction = "down" if y_diff > 0 else "up"

                        # 방향 업데이트
                        if self.object_movements[obj_id].get("movement_direction") is None:
                            self.object_movements[obj_id]["movement_direction"] = movement_direction

                    # 현재 위치 정보 추가
                    self.object_movements[obj_id]["trajectory"].append({
                        'center': current_center,
                        'bbox': (x, y, x + w, y + h),
                        'class_name': 'litter',
                        'confidence': 1.0
                    })

                    # 카운트 증가
                    self.object_movements[obj_id]["count"] = self.object_movements[obj_id].get("count", 0) + 1
                    self.object_movements[obj_id]["last_update"] = time.time()

            # DetectionStrategyManager를 이용한 체계적인 감지 로직
            if len(self.object_movements[obj_id]["trajectory"]) >= 2:
                # 카운트가 임계값을 넘었는지 확인
                count = self.object_movements[obj_id].get("count", 0)
                min_frames = self.config.min_frame_count_for_violation
                
                # 모든 객체에 대해 기본 디버깅 정보 출력 (임계값 관계없이)
                if self.config.debug_detection:
                    # 임계값 미달 객체는 간단한 정보만 출력
                    if count < min_frames:
                        print(f"\n🔍 [객체 추적 정보] ID: {obj_id}")
                        print(f"📊 프레임 카운트: {count}/{min_frames} (임계값 미달 - 전략 검사 스킵)")
                        print(f"📍 현재 위치: ({x + w // 2}, {y + h // 2})")
                        print(f"📏 크기: {w} x {h} (면적: {w*h} 픽셀)")
                        print(f"{'='*40}\n")
                
                # 카운트 임계값 미달 시 전략 검사 스킵
                if count < min_frames:
                    continue
                
                try:
                    # DetectionStrategyManager를 사용한 전략 검사
                    # 캐시 관리자를 config에 임시로 추가하여 전략들이 사용할 수 있도록 함
                    if hasattr(self, 'cache_manager'):
                        config_with_cache = type(self.config)()
                        # 기존 config 속성들 복사
                        for attr_name in dir(self.config):
                            if not attr_name.startswith('_'):
                                setattr(config_with_cache, attr_name, getattr(self.config, attr_name))
                        # 캐시 관리자 추가
                        config_with_cache.cache_manager = self.cache_manager
                        config_to_use = config_with_cache
                    else:
                        config_to_use = self.config
                    
                    strategy_results = self.strategy_manager.check_strategies(
                        frame=frame,
                        tracking_info=self.object_movements[obj_id]["trajectory"],
                        config=config_to_use,
                        vehicle_info=vehicle_info
                    )
                    
                    # Config의 detection_logic 설정에 따른 최종 판정
                    if self.config.detection_logic == "ANY":
                        # 하나라도 True면 성공
                        detection_result = any(strategy_results.values()) if strategy_results else False
                    elif self.config.detection_logic == "ALL":
                        # 모두 True면 성공
                        detection_result = all(strategy_results.values()) if strategy_results else False
                    else:
                        # 기본값: ALL
                        detection_result = all(strategy_results.values()) if strategy_results else False
                    
                    # 상세 디버깅 정보 출력 (임계값 충족 객체만)
                    self.debug_detection_info(obj_id, (x, y, w, h), detection_result, strategy_results)
                    
                    # 상세 전략 분석 출력 (검출 여부와 관계없이)
                    if self.config.debug_detection:
                        self.debug_strategy_details(obj_id, self.object_movements[obj_id]["trajectory"], vehicle_info)
                    
                    # 쓰레기 투기 감지 처리
                    if detection_result and not self.object_movements[obj_id].get("video_saved", False):
                        logger.info(f"🎯 쓰레기 투기 감지 성공: ID={obj_id}, 프레임 카운트={count}")
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        
                        # 🔥 핵심: 감지 완료 플래그 설정 (더 이상 전략 검사하지 않음)
                        self.object_movements[obj_id]["detection_completed"] = True
                        self.object_movements[obj_id]["video_saved"] = True
                        
                        # 차량과 위반 연결
                        self.link_violation_with_vehicle(obj_id, current_center, w)
                        
                        # 쓰레기 투기 감지 신호 발생
                        bbox = (x, y, x + w, y + h)
                        class_name = "litter"
                        confidence = 1.0
                        self.littering_detected_signal.emit(bbox, class_name, confidence, strategy_results)
                        
                        # 이벤트 활성화 (중복 방지)
                        with self.event_lock:
                            if not self.event_active:
                                self.event_triggered_at = self.current_frame_index
                                self.event_active = True
                                logger.info(f"쓰레기 투기 이벤트 활성화: 프레임 #{self.event_triggered_at}")
                                
                                # 이벤트 활성화 직후 즉시 한 번 호출 (현재 프레임에 대해)
                                self.collect_post_event_frame(frame, roi_x1, roi_y1)
                                
                except Exception as e:
                    logger.error(f"전략 검사 중 오류 (객체 ID: {obj_id}): {str(e)}")
                    logger.error(traceback.format_exc())
                    # 오류 발생 시 기본 로직으로 폴백
                    strategy_results = {"error": True}
                    self.debug_detection_info(obj_id, (x, y, w, h), False, strategy_results)

    def check_gravity_direction(self, trajectory):
        """궤적에서 하강 움직임을 확인하는 함수"""
        if len(trajectory) < 2:
            return False

        # 최근 몇 개의 포인트에서 y 좌표 변화 확인
        points_to_check = min(5, len(trajectory))
        positions = [info['center'] for info in trajectory[-points_to_check:]]

        # y 좌표가 증가하는지 확인 (하강 움직임)
        falling_count = 0
        for i in range(1, len(positions)):
            if positions[i][1] > positions[i - 1][1]:  # y 좌표가 증가 (하강)
                falling_count += 1

        # 80% 이상의 이동이 하강이면 True
        return falling_count >= int(0.8 * (len(positions) - 1))

    def check_vehicle_distance(self, obj_center, vehicle_info):
        """객체가 차량 근처에 있는지 확인하는 함수 (고성능 캐시 최적화 적용)"""
        if not vehicle_info and hasattr(self, 'cache_manager'):
            # 차량 정보가 없을 때 캐시에서 조회
            current_frame = getattr(self, 'current_frame_index', 0)
            return self.cache_manager.check_vehicle_distance(
                obj_center=obj_center,
                max_distance=self.config.distance_trash,
                frame_index=current_frame
            )
        elif vehicle_info:
            # 차량 정보가 있을 때는 고성능 캐시를 사용한 거리 계산
            min_distance = float('inf')
            
            if hasattr(self, 'cache_manager'):
                # 고성능 캐시를 사용한 거리 계산
                for vehicle in vehicle_info:
                    veh_center = vehicle['center']
                    dist = self.cache_manager.get_optimized_distance(obj_center, veh_center)
                    min_distance = min(min_distance, dist)
            else:
                # 기존 방식의 거리 계산
                for vehicle in vehicle_info:
                    veh_center = vehicle['center']
                    dist = math.sqrt((obj_center[0] - veh_center[0]) ** 2 + (obj_center[1] - veh_center[1]) ** 2)
                    min_distance = min(min_distance, dist)
        else:
            return False

        # 설정된 거리 임계값보다 가까우면 True
        return min_distance <= self.config.distance_trash

    def get_vehicle_info_from_boxes(self, boxes):
        """yolo_boxes 형식을 차량 정보 딕셔너리 리스트로 변환"""
        vehicles = []
        for (x1, y1, x2, y2) in boxes:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            vehicles.append({
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y)
            })
        return vehicles

    def collect_post_event_frame(self, frame, roi_x, roi_y):
        """이벤트 발생 후 프레임 수집 및 저장 결정"""
        logger.debug(f"collect_post_event_frame 호출됨: event_active={self.event_active}")
        if not self.event_active:
            return

        # 이벤트 발생 이후 경과한 프레임 수 계산
        frames_since_event = self.current_frame_index - self.event_triggered_at

        # 목표 프레임 수 계산 (이벤트 이후 저장할 프레임 수)
        target_post_frames = int(self.fps * self.post_event_duration)

        # 타겟 프레임 수에 도달하거나 프레임이 없을 때(None 신호) 영상 저장 시작
        frame_is_none = frame is None
        target_reached = frames_since_event >= target_post_frames

        if target_reached or frame_is_none:
            if frame_is_none:
                logger.info("프레임이 None (동영상 종료) - 현재까지 수집된 프레임으로 영상 저장")
            else:
                logger.info(f"영상 저장 시작: {target_post_frames}프레임 모두 수집 완료")

            # 이벤트 활성화 상태 확인 및 비활성화
            with self.event_lock:
                # 중복 저장 방지
                if not self.event_active:
                    return

                # 이벤트 비활성화
                self.event_active = False

                # 버퍼에서 연속된 프레임 추출
                with self.buffer_lock:
                    # 현재 버퍼 내용 복사
                    buffer_frames = list(self.continuous_buffer)
                    logger.info(f"버퍼에서 추출한 프레임 수: {len(buffer_frames)}")

                # 이벤트 발생 시점 기준으로 저장할 프레임 범위 계산
                pre_event_frames = int(self.fps * self.buffer_duration)
                total_frames = len(buffer_frames)

                # 버퍼에서 현재 이벤트 위치 찾기
                event_position_in_buffer = total_frames - frames_since_event
                logger.info(
                    f"이벤트 위치: total_frames={total_frames}, event_position={event_position_in_buffer}, pre_event_frames={pre_event_frames}")

                # 이벤트 전 프레임과 이벤트 후 프레임 분리
                start_index = max(0, event_position_in_buffer - pre_event_frames)

                # 연속된 프레임을 추출하여 저장
                frames_to_save = buffer_frames[start_index:]
                logger.info(f"저장할 총 프레임 수: {len(frames_to_save)}")

                if len(frames_to_save) > 0:
                    logger.info(f"연속 영상 저장 시작: 총 {len(frames_to_save)} 프레임"
                                f" (이벤트 전 {min(pre_event_frames, event_position_in_buffer)} 프레임,"
                                f" 이벤트 후 {frames_since_event} 프레임)")

                    # 저장 요청
                    try:
                        self.write_queue.put((frames_to_save, []))
                        logger.info("영상 저장 요청 완료, write_queue에 추가됨")

                        # 비디오 경로 저장 처리를 위한 더미 신호
                        if hasattr(self, 'video_saved_signal'):
                            video_path = os.path.join(self.config.output_dir,
                                                      f"trash_event_{time.strftime('%Y%m%d_%H%M%S')}.avi")
                            self.video_saved_signal.emit(video_path)
                    except Exception as e:
                        logger.error(f"영상 저장 요청 중 오류: {str(e)}")
                else:
                    logger.error("저장할 프레임이 없습니다.")

    def parse_yolo_results(self, result, roi_x, roi_y):
        """
        YOLO 모델 결과를 파싱하여 차량 바운딩 박스를 반환합니다.

        Args:
            result: YOLO 모델의 결과 객체
            roi_x, roi_y: ROI 좌표 오프셋

        Returns:
            list: 차량 바운딩 박스 리스트 [(x1, y1, x2, y2), ...]
        """
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        yolo_boxes = []
        current_vehicles = {}

        for box, confidence, class_id in zip(boxes, confidences, classes):
            if confidence > self.config.yolo_confidence_value and int(class_id) in self.vehicle_classes:
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                if width < self.min_box_size or height < self.min_box_size:
                    continue
                # 바운딩박스를 약간 shrink
                shrink_amount = 20
                x1 += shrink_amount
                x2 -= shrink_amount
                # ROI 좌표계에서 원본 좌표계로 변환
                x1 += roi_x
                y1 += roi_y
                x2 += roi_x
                y2 += roi_y
                yolo_boxes.append((x1, y1, x2, y2))

                # 차량 추적용 중심
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                matched = False
                for vid, prev_center in self.vehicle_tracking.items():
                    dist = math.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
                    if dist < 100:
                        current_vehicles[vid] = center
                        matched = True
                        if vid in self.violation_vehicles:
                            self.detected_vehicles.add((x1, y1, x2, y2))
                        break
                if not matched:
                    new_id = len(self.vehicle_tracking) + 1 if not self.vehicle_tracking else max(
                        self.vehicle_tracking.keys()) + 1
                    current_vehicles[new_id] = center

        # 현재 프레임에 없는 차량의 위반 ID 제거
        self.violation_vehicles = {vid for vid in self.violation_vehicles if vid in current_vehicles}

        # 현재 프레임의 차량만 유지
        self.vehicle_tracking = current_vehicles
        
        # 캐시 관리자에 차량 정보 업데이트
        if hasattr(self, 'cache_manager'):
            self.cache_manager.update_vehicles(
                frame_index=getattr(self, 'current_frame_index', 0),
                yolo_boxes=yolo_boxes,
                roi_offset=(roi_x, roi_y)
            )

        return yolo_boxes

    def detect_litter_candidates(self, roi_frame, object_detector, roi_x, roi_y, yolo_boxes):
        """
        배경 차분을 통해 쓰레기 후보 객체를 검출하는 함수

        Args:
            roi_frame: ROI 영역 이미지
            object_detector: 배경 차분 객체
            roi_x, roi_y: ROI 좌표
            yolo_boxes: 차량 바운딩 박스 목록

        Returns:
            list: 검출된 객체 리스트 [x, y, w, h]
        """
        try:
            # 배경 차분 적용
            mask = object_detector.apply(roi_frame)

            # 임계값 적용
            _, mask = cv2.threshold(mask, 200, 250, cv2.THRESH_BINARY)

            # 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 윤곽선 검출
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 디버깅 로그
            logger.debug(f"배경 차분 후 검출된 윤곽선 수: {len(contours)}")

            contour_detections = []
            for cnt in contours:
                # 면적 계산
                area = cv2.contourArea(cnt)

                # 크기 필터링
                if self.min_size < area < self.max_size:
                    x, y, w, h = cv2.boundingRect(cnt)

                    # ROI 내의 좌표를 전체 프레임 좌표로 변환
                    x += roi_x
                    y += roi_y

                    # 차량 바운딩박스와 겹치면 쓰레기가 아닌 것으로 간주
                    overlap = False
                    overlap_ratio = 0

                    for bx1, by1, bx2, by2 in yolo_boxes:
                        # 바운딩 박스 면적
                        obj_area = w * h

                        # 겹치는 영역 계산
                        intersection_x1 = max(x, bx1)
                        intersection_y1 = max(y, by1)
                        intersection_x2 = min(x + w, bx2)
                        intersection_y2 = min(y + h, by2)

                        if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                            intersection_area = (intersection_x2 - intersection_x1) * (
                                        intersection_y2 - intersection_y1)
                            overlap_ratio = intersection_area / obj_area

                            # 겹침 비율이 임계값 이상이면 객체 제외
                            if overlap_ratio > 0.5:  # 임계값은 필요에 따라 조정
                                overlap = True
                                break

                    # 차량과 겹치지 않는 객체만 추가
                    if not overlap:
                        contour_detections.append([x, y, w, h])
                        logger.debug(f"쓰레기 후보 검출: 위치=({x},{y}), 크기={w}x{h}, 면적={area}")

            logger.debug(f"최종 검출된 쓰레기 후보 객체 수: {len(contour_detections)}")
            return contour_detections

        except Exception as e:
            logger.error(f"쓰레기 후보 검출 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def mark_vehicles(self, frame, yolo_boxes):
        """
        프레임에 차량을 표시하는 함수 (위반 차량은 빨간색으로 강조)

        Args:
            frame: 표시할 프레임
            yolo_boxes: 차량 바운딩 박스 목록
        """
        for (bx1, by1, bx2, by2) in yolo_boxes:
            is_violation = False
            box_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

            for vid, center in self.vehicle_tracking.items():
                dist = math.sqrt((box_center[0] - center[0]) ** 2 + (box_center[1] - center[1]) ** 2)
                if dist < 50 and vid in self.violation_vehicles:
                    is_violation = True
                    break

            if is_violation or (bx1, by1, bx2, by2) in self.detected_vehicles:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(frame, "Violation Vehicle", (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                cv2.putText(frame, "Vehicle", (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def link_violation_with_vehicle(self, obj_id, litter_center, litter_width):
        """
        쓰레기 투기 객체를 주변 차량과 연결하는 함수 (캐시 최적화 적용)

        Args:
            obj_id: 쓰레기 객체 ID
            litter_center: 쓰레기 객체 중심점
            litter_width: 쓰레기 객체 너비
        """
        for vid, center in self.vehicle_tracking.items():
            vehicle_left_mid = (center[0] - litter_width // 2, center[1])
            vehicle_right_mid = (center[0] + litter_width // 2, center[1])

            # 캐시를 사용한 거리 계산
            if hasattr(self, 'cache_manager'):
                distance1 = self.cache_manager.get_optimized_distance(litter_center, vehicle_left_mid)
                distance2 = self.cache_manager.get_optimized_distance(litter_center, vehicle_right_mid)
            else:
                # 기존 방식의 거리 계산
                distance1 = math.sqrt((litter_center[0] - vehicle_left_mid[0]) ** 2 +
                                      (litter_center[1] - vehicle_left_mid[1]) ** 2)
                distance2 = math.sqrt((litter_center[0] - vehicle_right_mid[0]) ** 2 +
                                      (litter_center[1] - vehicle_right_mid[1]) ** 2)

            if distance1 <= self.config.distance_trash or distance2 <= self.config.distance_trash:
                logger.info(f"Vehicle ID: {vid}, Distance1: {distance1}, Distance2: {distance2}")
                self.violation_vehicles.add(vid)

                # 딕셔너리 접근 방식으로 수정
                if obj_id in self.object_movements and isinstance(self.object_movements[obj_id], dict):
                    self.object_movements[obj_id]["video_saved"] = True
                break
