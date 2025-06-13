#ui.py
import sys
import os
import cv2
import json
import csv
import time
import logging
import traceback
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QWidget, QLabel, QSpinBox, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFileDialog, QMessageBox, QDialog, QCheckBox, QComboBox,
    QTabWidget, QRadioButton, QGridLayout, QTextEdit, QApplication
)

# 기존 processing 모듈에서 가져온 함수 및 클래스들
from processing import save_roi_settings, load_roi_settings, VideoThread, Config, logger

# TS-ANPR API 모듈 import (예외 처리 추가)
try:
    import anpr_api
    ANPR_AVAILABLE = True
    logger.info("TS-ANPR API 모듈 로드 성공")
except Exception as e:
    ANPR_AVAILABLE = False
    logger.error(f"TS-ANPR API 모듈 로드 실패: {str(e)}")

# ---------------------------
# 번호판인식 전용 QThread 클래스
# ---------------------------
class PlateRecognitionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    plate_recognized_signal = pyqtSignal(str, int, tuple)  # 번호판, 신뢰도, 좌표

    def __init__(self, video_path, roi_x, roi_y, roi_width, roi_height, output_csv="plate_results.csv", frame_skip=2, parent=None):
        super().__init__(parent)
        if not video_path:
            raise ValueError("비디오 파일 경로가 올바르지 않습니다.")
        self.video_path = video_path
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        self._run_flag = True
        self.output_csv = output_csv
        self.frame_skip = frame_skip
        self.processed_count = 0
        self.recognized_set = set()  # 중복 번호판 방지
        self.last_plate_coords = None  # 마지막 인식된 번호판 좌표 추가

        # TS‑ANPR 초기화
        self.api_error = None
        if ANPR_AVAILABLE:
            try:
                self.api_error = anpr_api.initialize()
                if self.api_error:
                    logger.error("TS-ANPR 초기화 오류: " + self.api_error.replace("‑", "-"))
                else:
                    logger.info("TS-ANPR 초기화 성공")
            except Exception as e:
                self.api_error = str(e)
                logger.error(f"TS-ANPR 초기화 중 예외 발생: {str(e)}")
        else:
            self.api_error = "TS-ANPR 모듈이 로드되지 않았습니다."
            logger.warning("TS-ANPR 모듈이 로드되지 않아 번호판 인식 기능이 제한됩니다.")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"번호판인식 스레드: 비디오 열기 실패 - {self.video_path}")
            return

        # 원본 해상도 확인 및 로깅 추가
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"비디오 원본 해상도: {original_width}x{original_height}")

        # CSV 파일 생성 또는 열기
        csv_exists = os.path.exists(self.output_csv)
        with open(self.output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not csv_exists:
                writer.writerow(["detection_time", "license_plate", "ocr_conf", "plate_conf"])

        frame_count = 0
        total_processed = 0

        # 다양한 인식 옵션 시도
        recognition_options = ["v", "vs", "vm", "vms"]
        current_option_index = 0
        current_option = recognition_options[current_option_index]

        # 번호판 추적용 딕셔너리
        plate_confidence = {}  # 각 번호판의 최고 신뢰도 저장
        plate_best_text = {}  # 각 번호판의 전체 텍스트 저장
        plate_positions = {}  # 각 번호판의 마지막 인식 위치 저장
        final_plates = set()  # ROI를 벗어난 최종 번호판 저장

        # 최종 번호판 정보 저장용 변수
        best_overall_plate_text = ""
        best_overall_confidence = 0
        best_overall_coords = None
        best_overall_image = None

        print("\n===== 번호판 인식 시작 =====")
        # 인식 중 생성된 임시 스크린샷 파일 목록
        temp_screenshot_files = []

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                print("\n===== 번호판 인식 종료 =====")
                break

            frame_count += 1

            # 프레임 스킵: 매 frame_skip 프레임마다 인식
            if frame_count % self.frame_skip != 0:
                # ROI 영역 표시 (빨간색 박스로 변경)
                cv2.rectangle(frame, (self.roi_x, self.roi_y),
                              (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                              (0, 0, 255), 2)  # 빨간색 박스

                # 이전에 인식된 번호판 바운딩박스가 있으면 표시
                if self.last_plate_coords:
                    x1, y1, x2, y2 = self.last_plate_coords
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 녹색으로 번호판 표시

                self.change_pixmap_signal.emit(frame)
                continue

            total_processed += 1

            # ROI 영역 추출
            if (self.roi_y >= 0 and self.roi_x >= 0 and
                    self.roi_y + self.roi_height <= frame.shape[0] and
                    self.roi_x + self.roi_width <= frame.shape[1]):
                roi_frame = frame[self.roi_y:self.roi_y + self.roi_height,
                            self.roi_x:self.roi_x + self.roi_width].copy()
            else:
                roi_frame = frame.copy()  # 프레임 전체 사용

            if roi_frame is None or roi_frame.size == 0:
                cv2.rectangle(frame, (self.roi_x, self.roi_y),
                              (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                              (0, 255, 0), 2)
                self.change_pixmap_signal.emit(frame)
                continue

            # 다양한 해상도 시도 (원본 및 축소본)
            resized_versions = [
                ("원본", roi_frame.copy()),
                ("640x480", cv2.resize(roi_frame, (640, 480))),
                ("800x600", cv2.resize(roi_frame, (800, 600)))
            ]

            recognized = False
            best_plate_text = ""
            best_ocr_conf = 0
            best_plate_conf = 0
            plate_x_position = 0  # 번호판 X 좌표 저장

            for size_name, img in resized_versions:
                if recognized:
                    break  # 이미 인식된 경우 더 이상 시도하지 않음

                # BGR을 RGB로 변환
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                try:
                    # ANPR 모듈 사용 가능한 경우에만 번호판 인식 시도
                    if ANPR_AVAILABLE and not self.api_error:
                        result_json = anpr_api.readPixelsFromArray(img_rgb, "json", current_option)
                    else:
                        # ANPR 모듈 사용 불가능한 경우 빈 결과 반환
                        result_json = "[]"
                        continue  # 다음 크기의 이미지로 넘어감

                    if not result_json or result_json == "[]":
                        continue

                    try:
                        result_data = json.loads(result_json)

                        # JSON 형식에 따라 처리 방식 변경
                        if isinstance(result_data, list):
                            # 번호판 인식 결과 처리 (배열 형태)
                            for item in result_data:
                                if isinstance(item, dict) and "text" in item:
                                    # 텍스트 필드가 있는 경우 (번호판 직접 인식)
                                    plate_text = item.get("text", "")
                                    ocr_conf = item.get("conf", {}).get("ocr", 0)
                                    plate_conf = item.get("conf", {}).get("plate", 0)
                                    area = item.get("area", {})
                                    x_pos = area.get("x", 0)

                                    if plate_text and ocr_conf > best_ocr_conf:
                                        best_plate_text = plate_text
                                        best_ocr_conf = ocr_conf
                                        best_plate_conf = plate_conf
                                        plate_x_position = x_pos
                                        recognized = True

                                elif isinstance(item, dict) and "licensePlate" in item:
                                    # licensePlate 필드가 있는 경우 (객체 인식 결과)
                                    for plate in item.get("licensePlate", []):
                                        plate_text = plate.get("text", "")
                                        ocr_conf = plate.get("conf", {}).get("ocr", 0)
                                        plate_conf = plate.get("conf", {}).get("plate", 0)
                                        area = plate.get("area", {})
                                        x_pos = area.get("x", 0)

                                        if plate_text and ocr_conf > best_ocr_conf:
                                            best_plate_text = plate_text
                                            best_ocr_conf = ocr_conf
                                            best_plate_conf = plate_conf
                                            plate_x_position = x_pos
                                            recognized = True

                        elif isinstance(result_data, dict):
                            # 오류 또는 직접 객체인 경우
                            if "licensePlate" in result_data:
                                # 직접 객체인 경우 처리
                                for plate in result_data.get("licensePlate", []):
                                    plate_text = plate.get("text", "")
                                    ocr_conf = plate.get("conf", {}).get("ocr", 0)
                                    plate_conf = plate.get("conf", {}).get("plate", 0)
                                    area = plate.get("area", {})
                                    x_pos = area.get("x", 0)

                                    if plate_text and ocr_conf > best_ocr_conf:
                                        best_plate_text = plate_text
                                        best_ocr_conf = ocr_conf
                                        best_plate_conf = plate_conf
                                        plate_x_position = x_pos
                                        recognized = True
                    except Exception as e:
                        pass  # 디버깅 창에 오류 표시 없음

                except Exception as e:
                    pass  # 디버깅 창에 오류 표시 없음

            # 인식 결과가 있는 경우 처리
            if recognized and best_plate_text:
                # 번호판 위치 좌표 계산
                plate_coords = None

                # 현재 처리 중인 item이 있는지 확인 (루프 바깥에서는 정의되지 않을 수 있음)
                if 'item' in locals() and isinstance(item, dict) and 'area' in item:
                    if all(k in item['area'] for k in ['x', 'y', 'width', 'height']):
                        # 좌표 추출 (ROI 좌표에 상대적)
                        x = item['area']['x'] + self.roi_x
                        y = item['area']['y'] + self.roi_y
                        w = item['area']['width']
                        h = item['area']['height']
                        plate_coords = (x, y, x + w, y + h)
                        self.last_plate_coords = plate_coords

                # 번호판 좌표가 있으면 표시
                if plate_coords:
                    x1, y1, x2, y2 = plate_coords
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 녹색으로 번호판 표시

                    # 인식된 번호판 위에 텍스트 표시
                    cv2.putText(frame, best_plate_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 전체 비디오에서 최고 신뢰도 번호판 업데이트
                confidence_percentage = int(best_ocr_conf * 100)
                if confidence_percentage > best_overall_confidence:
                    best_overall_plate_text = best_plate_text
                    best_overall_confidence = confidence_percentage
                    best_overall_coords = plate_coords if plate_coords else None
                    best_overall_image = frame.copy()  # 현재 프레임 복사

                # 끝 4자리 추출 (번호판 길이가 4글자 이상인 경우)
                plate_suffix = best_plate_text[-4:] if len(best_plate_text) >= 4 else best_plate_text

                # 해당 끝자리의 기존 신뢰도 확인
                current_best_conf = plate_confidence.get(plate_suffix, 0)

                # 현재 인식된 번호판의 신뢰도가 더 높은 경우 업데이트
                if best_ocr_conf > current_best_conf:
                    plate_confidence[plate_suffix] = best_ocr_conf
                    plate_best_text[plate_suffix] = best_plate_text
                    # 인식 결과 시그널 발생 - 이 부분 추가
                    confidence_percentage = int(best_ocr_conf * 100)
                    self.plate_recognized_signal.emit(best_plate_text, confidence_percentage,
                                                      plate_coords if plate_coords else (0, 0, 0, 0))

                    # 이미지 저장 - 임시 스크린샷으로 저장
                    if plate_suffix not in self.recognized_set:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"temp_plate_screenshot_{best_plate_text}_{timestamp}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        temp_screenshot_files.append(screenshot_path)  # 임시 파일 목록에 추가
                        logging.info(f"임시 번호판 스크린샷 저장: {screenshot_path}")

                        # 현재 시간 생성
                        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # CSV에 결과 추가
                        with open(self.output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([detection_time, best_plate_text, best_ocr_conf, best_plate_conf])

                        # 디버깅 창에 간결하게 표시
                        confidence_percentage = int(best_ocr_conf * 100)
                        print(f"인식된 번호판: {best_plate_text} (신뢰도: {confidence_percentage}%)")

                        # 인식된 번호판 추가
                        self.recognized_set.add(plate_suffix)

                # 번호판 위치 업데이트
                plate_positions[plate_suffix] = plate_x_position

                # 번호판이 ROI 왼쪽 경계에 가까워지면 최종 번호로 판정
                roi_left_boundary = 0  # ROI 내 왼쪽 경계 (0 = ROI 시작점)
                if plate_x_position <= roi_left_boundary and plate_suffix not in final_plates:
                    # 최종 번호판으로 판별
                    final_plates.add(plate_suffix)
                    final_plate_text = plate_best_text[plate_suffix]
                    final_confidence = plate_confidence[plate_suffix]
                    confidence_percentage = int(final_confidence * 100)
                    print(f"최종 번호판: {final_plate_text} (신뢰도: {confidence_percentage}%)")
                    # 최종 번호판 시그널 발생 - 이 부분 수정
                    self.plate_recognized_signal.emit(final_plate_text, confidence_percentage,
                                                      self.last_plate_coords if self.last_plate_coords else (
                                                      0, 0, 0, 0))

            # ROI 영역만 표시 (번호 표시 없음)
            cv2.rectangle(frame, (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                          (0, 0, 255), 2)  # 빨간색 박스로 변경

            self.change_pixmap_signal.emit(frame)
            time.sleep(0.03)  # UI 업데이트 시간 여유

        cap.release()

        # 비디오 처리 완료 후, 최종 번호판 저장
        if best_overall_image is not None and best_overall_plate_text:
            # 최종 결과 이미지에 바운딩 박스 표시
            if best_overall_coords:
                x1, y1, x2, y2 = best_overall_coords
                cv2.rectangle(best_overall_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 바운딩 박스

                # 이미지 상단에 최종 번호판 텍스트 표시
                cv2.putText(best_overall_image,
                            f"최종 번호판: {best_overall_plate_text} (신뢰도: {best_overall_confidence}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 최종 이미지 저장 (이름에 번호판 정보 포함)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_result_path = f"final_plate_{best_overall_plate_text}_{best_overall_confidence}pct_{timestamp}.jpg"
            cv2.imwrite(final_result_path, best_overall_image)
            print(f"최종 번호판 이미지 저장됨: {final_result_path}")
            logging.info(f"최종 번호판 이미지 저장: {final_result_path} (신뢰도: {best_overall_confidence}%)")

            # 모든 임시 스크린샷 파일 삭제
            for temp_file in temp_screenshot_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logging.debug(f"임시 파일 삭제됨: {temp_file}")
                except Exception as e:
                    logging.error(f"임시 파일 삭제 중 오류: {temp_file} - {str(e)}")

            logging.info(f"{len(temp_screenshot_files)}개의 임시 스크린샷이 정리되었습니다.")

            # 최종 결과를 UI에 보고
            self.plate_recognized_signal.emit(
                best_overall_plate_text,
                best_overall_confidence,
                best_overall_coords if best_overall_coords else (0, 0, 0, 0)
            )

        print(f"\n총 인식된 번호판: {len(plate_best_text)}개")
        if plate_best_text:
            print("인식된 번호판 목록:")
            for idx, (suffix, text) in enumerate(plate_best_text.items(), 1):
                confidence = int(plate_confidence[suffix] * 100)
                is_final = "최종" if suffix in final_plates else ""
                print(f"  {idx}. {text} (신뢰도: {confidence}%) {is_final}")


class SettingsDialog(QDialog):
    """설정 창"""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setGeometry(150, 150, 450, 550)  # 높이를 약간 늘려 줌

        # 설정 값을 임시 저장할 변수들
        self.temp_settings = {}

        if config is None:
            logging.error("설정 창 초기화 실패: config 객체가 None입니다.")
            QMessageBox.critical(self, "Error", "설정을 불러올 수 없습니다.")
            self.reject()
            return

        self.config = config

        # 현재 설정을 임시 변수에 저장
        self.temp_settings['min_size'] = self.config.min_size
        self.temp_settings['max_size'] = self.config.max_size
        self.temp_settings['yolo_confidence'] = int(self.config.yolo_confidence_value * 100)
        self.temp_settings['batch_size'] = self.config.batch_size
        self.temp_settings['gravity_direction_threshold'] = self.config.gravity_direction_threshold
        self.temp_settings['distance_trash'] = self.config.distance_trash
        self.temp_settings['detection_logic'] = self.config.detection_logic
        self.temp_settings['horizontal_direction_threshold'] = getattr(self.config, 'horizontal_direction_threshold', 5)
        self.temp_settings['max_vehicle_distance'] = getattr(self.config, 'max_vehicle_distance', 200)
        self.temp_settings['debug_detection'] = self.config.debug_detection

        # UI 생성
        self.setup_ui()

    def setup_ui(self):
        """UI 구성 요소 설정"""
        layout = QVBoxLayout()

        # 탭 위젯 추가
        tab_widget = QTabWidget()

        # 기본 설정 탭
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()

        # Min Size 설정
        size_group = QGroupBox("객체 크기 설정")
        size_layout = QGridLayout()

        # Min Size
        self.min_size_spinbox = QSpinBox(self)
        self.min_size_spinbox.setRange(1, 1000)
        self.min_size_spinbox.setValue(self.temp_settings['min_size'])
        self.min_size_spinbox.setToolTip("쓰레기 객체의 최소 크기 (픽셀 단위)")
        size_layout.addWidget(QLabel("Min Size:"), 0, 0)
        size_layout.addWidget(self.min_size_spinbox, 0, 1)

        # Max Size 설정
        self.max_size_spinbox = QSpinBox(self)
        self.max_size_spinbox.setRange(1, 5000)
        self.max_size_spinbox.setValue(self.temp_settings['max_size'])
        self.max_size_spinbox.setToolTip("쓰레기 객체의 최대 크기 (픽셀 단위)")
        size_layout.addWidget(QLabel("Max Size:"), 1, 0)
        size_layout.addWidget(self.max_size_spinbox, 1, 1)

        size_group.setLayout(size_layout)
        basic_layout.addWidget(size_group)

        # YOLO 및 배치 설정 그룹
        yolo_group = QGroupBox("YOLO 설정")
        yolo_layout = QGridLayout()

        # YOLO Confidence 설정
        self.yolo_confidence_spinbox = QSpinBox(self)
        self.yolo_confidence_spinbox.setRange(1, 100)
        self.yolo_confidence_spinbox.setValue(self.temp_settings['yolo_confidence'])
        self.yolo_confidence_spinbox.setToolTip("객체 검출 신뢰도 임계값 (%)")
        yolo_layout.addWidget(QLabel("YOLO Confidence (%):"), 0, 0)
        yolo_layout.addWidget(self.yolo_confidence_spinbox, 0, 1)

        # Batch Size 콤보박스
        self.batch_size_combobox = QComboBox(self)
        batch_sizes = [1, 4, 8, 16, 32]
        for bs in batch_sizes:
            self.batch_size_combobox.addItem(str(bs))

        # 현재 config.batch_size 값을 기본으로 설정
        try:
            current_index = batch_sizes.index(self.temp_settings['batch_size']) if self.temp_settings[
                                                                                       'batch_size'] in batch_sizes else 2
            self.batch_size_combobox.setCurrentIndex(current_index)
        except (ValueError, IndexError) as e:
            logging.warning(f"배치 크기 설정 중 오류: {str(e)}, 기본값 8로 설정됩니다.")
            self.batch_size_combobox.setCurrentIndex(2)  # 기본값 8 (인덱스 2)

        self.batch_size_combobox.setToolTip("YOLO 모델 배치 처리 크기")
        yolo_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        yolo_layout.addWidget(self.batch_size_combobox, 1, 1)

        yolo_group.setLayout(yolo_layout)
        basic_layout.addWidget(yolo_group)

        # Distance Trash 설정 (기본 설정 탭으로 이동)
        distance_group = QGroupBox("거리 설정")
        distance_layout = QGridLayout()

        self.distance_trash_spinbox = QSpinBox(self)
        self.distance_trash_spinbox.setRange(1, 9999)
        self.distance_trash_spinbox.setValue(self.temp_settings['distance_trash'])
        self.distance_trash_spinbox.setToolTip("차량과 쓰레기 객체 간 최대 거리 (픽셀 단위)")
        distance_layout.addWidget(QLabel("Distance Trash:"), 0, 0)
        distance_layout.addWidget(self.distance_trash_spinbox, 0, 1)

        distance_group.setLayout(distance_layout)
        basic_layout.addWidget(distance_group)

        basic_tab.setLayout(basic_layout)
        tab_widget.addTab(basic_tab, "기본 설정")

        # 쓰레기 감지 전략 설정 탭
        strategy_tab = QWidget()
        strategy_layout = QVBoxLayout()

        # 감지 로직 선택 (ANY/ALL)
        logic_group = QGroupBox("감지 로직")
        logic_layout = QVBoxLayout()

        self.any_logic_radio = QRadioButton("어느 하나라도 조건 충족 (OR)")
        self.all_logic_radio = QRadioButton("모든 조건 충족 (AND)")

        if self.temp_settings['detection_logic'] == "ALL":
            self.all_logic_radio.setChecked(True)
        else:
            self.any_logic_radio.setChecked(True)

        logic_layout.addWidget(self.any_logic_radio)
        logic_layout.addWidget(self.all_logic_radio)
        logic_group.setLayout(logic_layout)
        strategy_layout.addWidget(logic_group)

        # 감지 임계값 설정
        thresholds_group = QGroupBox("감지 임계값 설정")
        thresholds_layout = QGridLayout()

        # Gravity Direction Threshold 설정
        self.gravity_direction_spinbox = QSpinBox(self)
        self.gravity_direction_spinbox.setRange(1, 50)
        self.gravity_direction_spinbox.setValue(self.temp_settings['gravity_direction_threshold'])
        self.gravity_direction_spinbox.setToolTip("중력 방향 이동 임계값 (픽셀 단위)")
        thresholds_layout.addWidget(QLabel("중력 방향 임계값:"), 0, 0)
        thresholds_layout.addWidget(self.gravity_direction_spinbox, 0, 1)

        # 수평 이동 임계값
        self.horizontal_direction_spinbox = QSpinBox(self)
        self.horizontal_direction_spinbox.setRange(1, 50)
        self.horizontal_direction_spinbox.setValue(self.temp_settings['horizontal_direction_threshold'])
        self.horizontal_direction_spinbox.setToolTip("수평 이동 임계값 (픽셀 단위)")
        thresholds_layout.addWidget(QLabel("수평 이동 임계값:"), 1, 0)
        thresholds_layout.addWidget(self.horizontal_direction_spinbox, 1, 1)

        # 최대 차량 거리
        self.max_vehicle_distance_spinbox = QSpinBox(self)
        self.max_vehicle_distance_spinbox.setRange(1, 1000)
        self.max_vehicle_distance_spinbox.setValue(self.temp_settings['max_vehicle_distance'])
        self.max_vehicle_distance_spinbox.setToolTip("차량과 오브젝트 간 최대 거리 (픽셀 단위)")
        thresholds_layout.addWidget(QLabel("최대 차량 거리:"), 2, 0)
        thresholds_layout.addWidget(self.max_vehicle_distance_spinbox, 2, 1)

        thresholds_group.setLayout(thresholds_layout)
        strategy_layout.addWidget(thresholds_group)

        strategy_tab.setLayout(strategy_layout)
        tab_widget.addTab(strategy_tab, "감지 전략")

        layout.addWidget(tab_widget)

        # 디버깅 설정 그룹박스 (버튼 위로 이동)
        debug_group = QGroupBox("디버깅 설정")
        debug_layout = QVBoxLayout()

        # 객체 검출 디버깅 체크박스
        self.debug_detection_checkbox = QCheckBox("객체 검출 디버깅 활성화")
        self.debug_detection_checkbox.setChecked(self.temp_settings['debug_detection'])
        self.debug_detection_checkbox.setToolTip("체크하면 객체 좌표와 검출 결과를 콘솔에 출력합니다.")
        debug_layout.addWidget(self.debug_detection_checkbox)

        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)

        # 확인 및 취소 버튼
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def save_settings(self):
        """설정값을 안전하게 저장하고 대화상자 종료"""
        try:
            # 임시 변수에서 설정값 가져와 설정
            self.config.min_size = self.min_size_spinbox.value()
            self.config.max_size = self.max_size_spinbox.value()
            self.config.yolo_confidence_value = self.yolo_confidence_spinbox.value() / 100.0
            self.config.batch_size = int(self.batch_size_combobox.currentText())
            self.config.distance_trash = self.distance_trash_spinbox.value()

            # 감지 로직 설정
            if self.all_logic_radio.isChecked():
                self.config.detection_logic = "ALL"
            else:
                self.config.detection_logic = "ANY"

            # 임계값 설정 저장
            self.config.gravity_direction_threshold = self.gravity_direction_spinbox.value()

            # 새로운 전략 관련 설정 저장
            self.config.horizontal_direction_threshold = self.horizontal_direction_spinbox.value()
            self.config.max_vehicle_distance = self.max_vehicle_distance_spinbox.value()

            # 디버깅 설정 저장
            self.config.debug_detection = self.debug_detection_checkbox.isChecked()

            logger.info(f"설정값 업데이트: min_size={self.config.min_size}, max_size={self.config.max_size}, "
                        f"yolo_confidence={self.config.yolo_confidence_value}, "
                        f"gravity_threshold={self.config.gravity_direction_threshold}, "
                        f"distance_trash={self.config.distance_trash}, "
                        f"batch_size={self.config.batch_size}, "
                        f"detection_logic={self.config.detection_logic}, "
                        f"horizontal_threshold={self.config.horizontal_direction_threshold}, "
                        f"max_vehicle_distance={self.config.max_vehicle_distance}, "
                        f"debug_detection={self.config.debug_detection}")

            self.accept()
        except Exception as e:
            logging.error(f"설정 저장 중 오류 발생: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"설정을 저장하는 중 오류가 발생했습니다: {str(e)}")


# ---------------------------
# 수정된 DetectionApp 클래스
# ---------------------------
class DetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LitteringDetection_v1.0")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.config = Config()  # Config 객체 초기화
        self.current_video_index = 0  # 초기화 추가
        self.processing_enabled = False  # 전체 프로세싱 제어 플래그

        # ROI 그리기 상태 관리 변수들 추가
        self.is_roi_drawing = False
        self.is_plate_roi_drawing = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.plate_roi_start_x = 0
        self.plate_roi_start_y = 0

        # ROI 설정값 불러오기
        try:
            self.roi_x, self.roi_y, self.roi_width, self.roi_height, self.min_size, self.max_size = load_roi_settings()
            logger.info(
                f"초기화된 ROI 값: x={self.roi_x}, y={self.roi_y}, width={self.roi_width}, height={self.roi_height}, "
                f"min_size={self.min_size}, max_size={self.max_size}"
            )
            # ROI 설정에서 로드된 min_size, max_size는 지역변수로만 사용
            # Config 객체는 default_settings.txt에서 로드된 값을 유지
        except Exception as e:
            logger.error(f"ROI 설정 로드 실패: {str(e)}")
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = 0, 300, 800, 150
            QMessageBox.warning(self, "Warning", "ROI 설정을 불러오는데 실패했습니다. 기본값을 사용합니다.")

        # 번호판 ROI 설정값 불러오기
        try:
            self.plate_roi_x, self.plate_roi_y, self.plate_roi_width, self.plate_roi_height = self.load_plate_roi_settings()
            logger.info(
                f"초기화된 번호판 ROI 값: x={self.plate_roi_x}, y={self.plate_roi_y}, "
                f"width={self.plate_roi_width}, height={self.plate_roi_height}"
            )
        except Exception as e:
            logger.error(f"번호판 ROI 설정 로드 실패: {str(e)}")
            self.plate_roi_x, self.plate_roi_y, self.plate_roi_width, self.plate_roi_height = 0, 100, 400, 150
            logger.warning("번호판 ROI 설정을 불러오는데 실패했습니다. 기본값을 사용합니다.")

        # 상태 변수 초기화
        self.image = None
        self.temp_image = None
        self.video_path = None
        self.drawing = False  # 마우스 드래그 상태
        self.detection_running = False  # 감지 동작 상태 플래그
        self.thread = None  # VideoThread 또는 PlateRecognitionThread 참조

        # 번호판 처리를 위한 변수
        self.saved_video_path = None  # 저장된 동영상 경로
        self.plate_recognition_thread = None  # 번호판 인식 스레드

        # UI 요소 생성
        self.init_ui()
        # TS-ANPR 초기화 상태 확인
        self.check_anpr_initialization()

    def save_strategy_settings(self):
        """전략 설정 저장"""
        try:
            settings = {
                "detection_logic": self.config.detection_logic,
                "gravity_direction_threshold": self.config.gravity_direction_threshold,
                "horizontal_direction_threshold": self.config.horizontal_direction_threshold,
                "vehicle_overlap_threshold": self.config.vehicle_overlap_threshold,
                "max_vehicle_distance": self.config.max_vehicle_distance,
                "distance_trash": self.config.distance_trash
            }

            # 활성화된 전략 ID 저장
            if hasattr(self, 'thread') and hasattr(self.thread, 'strategy_manager'):
                settings["enabled_strategies"] = list(self.thread.strategy_manager.enabled_strategies)

            with open("strategy_settings.json", "w") as f:
                json.dump(settings, f, indent=4)

            logging.info("전략 설정 저장 완료")

        except Exception as e:
            logging.error(f"전략 설정 저장 중 오류: {str(e)}")
            logging.error(traceback.format_exc())

    def validate_roi(self, frame_width=1920, frame_height=1080):
        """
        개선된 ROI 검증 함수 - 안전한 경계 검사 및 사용자 피드백
        """
        # ROI 끝 좌표 계산
        roi_end_x = self.roi_x + self.roi_width
        roi_end_y = self.roi_y + self.roi_height

        # ROI가 프레임 경계를 벗어나는지 확인
        is_valid = True
        error_messages = []

        if self.roi_x < 0:
            error_messages.append("ROI X 시작점이 0보다 작습니다.")
            is_valid = False
            
        if self.roi_y < 0:
            error_messages.append("ROI Y 시작점이 0보다 작습니다.")
            is_valid = False
            
        if roi_end_x > frame_width:
            error_messages.append(f"ROI가 프레임 너비({frame_width})를 초과합니다. (현재: {roi_end_x})")
            is_valid = False
            
        if roi_end_y > frame_height:
            error_messages.append(f"ROI가 프레임 높이({frame_height})를 초과합니다. (현재: {roi_end_y})")
            is_valid = False
            
        if self.roi_width <= 10:
            error_messages.append("ROI 너비가 너무 작습니다. (최소 10픽셀)")
            is_valid = False
            
        if self.roi_height <= 10:
            error_messages.append("ROI 높이가 너무 작습니다. (최소 10픽셀)")
            is_valid = False

        # 검증 실패 시 사용자에게 안전하게 알림
        if not is_valid:
            QTimer.singleShot(0, lambda: self.show_roi_validation_error(error_messages, frame_width, frame_height))
            return False

        return is_valid

    def show_roi_validation_error(self, error_messages, frame_width, frame_height):
        """ROI 검증 실패 시 안전하게 오류 메시지 표시"""
        error_text = "\n".join(error_messages)
        full_message = (
            f"ROI 설정에 문제가 있습니다:\n\n"
            f"{error_text}\n\n"
            f"프레임 크기: {frame_width}x{frame_height}\n"
            f"현재 ROI: (x={self.roi_x}, y={self.roi_y}, width={self.roi_width}, height={self.roi_height})\n\n"
            f"ROI를 다시 설정해주세요."
        )
        
        QMessageBox.warning(self, "ROI 설정 오류", full_message, QMessageBox.Ok)
        self.highlight_roi_button()

    def highlight_roi_button(self):
        """ROI 설정 버튼을 강조 표시"""
        try:
            # 빨간색으로 강조
            self.set_roi_button.setStyleSheet("""
                QPushButton {
                    font: bold 12pt Arial;
                    color: white;
                    background-color: #FF5733;
                    border: 2px solid #FF0000;
                    border-radius: 5px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #FF0000;
                }
            """)
            
            # 3번 깜빡이는 효과
            def blink_effect(count=0):
                if count < 6:
                    self.set_roi_button.setVisible(count % 2 == 0)
                    QApplication.processEvents()
                    QTimer.singleShot(300, lambda: blink_effect(count + 1))
                else:
                    self.set_roi_button.setVisible(True)
                    self.set_roi_button.setStyleSheet(self._button_style())
            
            blink_effect()
            
        except Exception as e:
            logger.error(f"ROI 버튼 강조 중 오류: {str(e)}")

    def check_anpr_initialization(self):
        """TS-ANPR 라이브러리 초기화 상태를 확인하고 UI에 표시"""
        try:
            # 상태 라벨이 없다면 생성
            if not hasattr(self, 'anpr_status_label'):
                self.anpr_status_label = QLabel("TS-ANPR 상태: 확인 중...", self)
                self.anpr_status_label.setStyleSheet("color: blue; font-weight: bold;")
                self.anpr_status_label.setFixedHeight(20)

                # 상태 라벨을 레이아웃에 추가
                if hasattr(self.layout(), 'addWidget'):
                    self.layout().addWidget(self.anpr_status_label)

            # TS-ANPR 초기화 상태 확인
            error = anpr_api.initialize()

            if error:
                # 초기화 실패
                logger.error(f"TS-ANPR 초기화 오류: {error}")
                self.anpr_status_label.setText(f"TS-ANPR 상태: 초기화 실패 ({error})")
                self.anpr_status_label.setStyleSheet("color: red; font-weight: bold;")

                # 사용자에게 알림
                QMessageBox.warning(
                    self,
                    "TS-ANPR 초기화 실패",
                    f"번호판 인식 엔진(TS-ANPR) 초기화에 실패했습니다.\n오류: {error}\n\n번호판 인식 기능이 동작하지 않을 수 있습니다."
                )
            else:
                # 초기화 성공
                logger.info("TS-ANPR 초기화 성공")
                self.anpr_status_label.setText("TS-ANPR 상태: 초기화 성공")
                self.anpr_status_label.setStyleSheet("color: green; font-weight: bold;")
        except Exception as e:
            logger.error(f"TS-ANPR 상태 확인 중 오류: {str(e)}")
            logger.error(traceback.format_exc())

            if hasattr(self, 'anpr_status_label'):
                self.anpr_status_label.setText(f"TS-ANPR 상태: 확인 오류 ({str(e)})")
                self.anpr_status_label.setStyleSheet("color: red; font-weight: bold;")

    def init_ui(self):
        try:
            # 이미지 레이블 생성
            self.image_label = QLabel(self)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #ffffff;")
            self.image_label.resize(1920, 1080)

            # 버튼 생성
            self.open_file_button = QPushButton("Open File")
            self.open_file_button.setStyleSheet(self._button_style())
            self.open_file_button.clicked.connect(self.open_single_file)

            self.select_folder_button = QPushButton("Select Folder")
            self.select_folder_button.setStyleSheet(self._button_style())
            self.select_folder_button.clicked.connect(self.open_file_dialog)

            self.set_roi_button = QPushButton("쓰레기 ROI 설정")
            self.set_roi_button.setStyleSheet(self._button_style())
            self.set_roi_button.setEnabled(False)
            self.set_roi_button.clicked.connect(self.start_drawing)

            # 번호판 ROI 설정 버튼 추가
            self.set_plate_roi_button = QPushButton("번호판 ROI 설정")
            self.set_plate_roi_button.setStyleSheet(self._plate_roi_button_style())  # 빨간색 스타일
            self.set_plate_roi_button.setEnabled(False)
            self.set_plate_roi_button.clicked.connect(self.start_plate_roi_drawing)

            self.run_button = QPushButton("Run Detection")
            self.run_button.setStyleSheet(self._run_button_style())
            self.run_button.setEnabled(False)
            self.run_button.clicked.connect(self.start_detection)

            self.stop_button = QPushButton("Stop Detection")
            self.stop_button.setStyleSheet(self._button_style())
            self.stop_button.setEnabled(False)
            self.stop_button.clicked.connect(self.stop_detection)

            self.settings_button = QPushButton("⚙")
            self.settings_button.setFixedSize(40, 50)
            self.settings_button.setStyleSheet(self._settings_button_style())
            self.settings_button.clicked.connect(self.open_settings_dialog)

            # ---- 추가된 번호판인식 체크박스 ----
            self.plate_recognition_checkbox = QCheckBox("번호판인식")
            self.plate_recognition_checkbox.setChecked(False)
            self.plate_recognition_checkbox.setToolTip("체크 시 쓰레기 투기 검출 이후 번호판인식 알고리즘이 자동으로 실행됩니다.")

            self._setup_layout()
        except Exception as e:
            logger.error(f"UI 초기화 중 오류 발생: {str(e)}")
            QMessageBox.critical(self, "Error", f"UI 초기화 중 오류가 발생했습니다: {str(e)}")

    def _setup_layout(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 이미지 레이블
        self.image_label.setMinimumHeight(500)
        self.image_label.setFixedSize(1900, 950)
        main_layout.addWidget(self.image_label)

        # 버튼 그룹
        button_layout = QHBoxLayout()
        self.run_button.setStyleSheet(self._run_button_style())

        # File Operations 그룹
        file_operation_group = QGroupBox("File Operations")
        file_operation_layout = QHBoxLayout()
        file_operation_layout.setContentsMargins(5, 5, 5, 5)
        file_operation_layout.addWidget(self.select_folder_button)
        file_operation_layout.addWidget(self.open_file_button)
        file_operation_layout.addWidget(self.run_button)
        file_operation_layout.addWidget(self.stop_button)
        file_operation_group.setLayout(file_operation_layout)

        # Settings 그룹 (추가: 번호판인식 체크박스 포함)
        settings_group = QGroupBox("Settings")
        settings_group.setFixedHeight(80)
        settings_group.setStyleSheet("QGroupBox { font-size: 10pt; }")
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(5, 5, 5, 5)
        self.set_roi_button.setFixedSize(120, 30)
        self.set_plate_roi_button.setFixedSize(120, 30)  # 번호판 ROI 버튼 크기
        self.settings_button.setFixedSize(80, 30)

        # 버튼과 체크박스를 함께 배치
        settings_layout.addWidget(self.set_roi_button)
        settings_layout.addWidget(self.set_plate_roi_button)  # 번호판 ROI 버튼 추가
        settings_layout.addWidget(self.settings_button)
        settings_layout.addWidget(self.plate_recognition_checkbox)
        settings_group.setLayout(settings_layout)

        button_layout.addWidget(file_operation_group, 3)
        button_layout.addWidget(settings_group, 1)
        main_layout.addLayout(button_layout)
        # TS-ANPR 상태 표시 라벨을 위한 레이아웃 추가
        status_layout = QHBoxLayout()
        self.anpr_status_label = QLabel("TS-ANPR 상태: 확인 중...", self)
        self.anpr_status_label.setStyleSheet("color: blue; font-weight: bold;")
        status_layout.addWidget(self.anpr_status_label)
        status_layout.addStretch()
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

    def open_settings_dialog(self):
        try:
            from processing import update_config
            dialog = SettingsDialog(self, config=self.config)
            if dialog.exec_():
                QMessageBox.information(self, "Settings", "설정이 저장되었습니다!")
                logger.info(f"설정값: min_size={self.config.min_size}, max_size={self.config.max_size}, "
                            f"yolo_confidence={self.config.yolo_confidence_value}, "
                            f"gravity_direction_threshold={self.config.gravity_direction_threshold}, "
                            f"batch_size={self.config.batch_size}")
                update_config(self.config)
        except Exception as e:
            logger.error(f"설정 창 열기 중 오류: {str(e)}")
            QMessageBox.critical(self, "Error", f"설정 창을 열 수 없습니다: {str(e)}")

    def _plate_roi_button_style(self):
        return (
            "QPushButton {"
            "  font: bold 10pt Arial;"
            "  color: white;"
            "  background-color: #FF0000;"  # 빨간색
            "  border: 1px solid #CC0000;"
            "  border-radius: 5px;"
            "  padding: 8px 16px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #CC0000;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #cccccc;"
            "  color: #666666;"
            "}"
        )
    def _button_style(self):
        return (
            "QPushButton {"
            "  font: bold 10pt Arial;"
            "  color: white;"
            "  background-color: #007BFF;"
            "  border: 1px solid #0056b3;"
            "  border-radius: 5px;"
            "  padding: 8px 16px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #0056b3;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #cccccc;"
            "  color: #666666;"
            "}"
        )

    def _run_button_style(self):
        return """
            QPushButton {
                font: bold 10pt Arial;
                color: white;
                background-color: #FF5733;
                border: 1px solid #B93C1D;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #E74C3C;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """

    def _settings_button_style(self):
        return (
            "QPushButton {"
            "  font: bold 14pt Arial;"
            "  color: black;"
            "  background-color: #f0f0f0;"
            "  border: 1px solid #ccc;"
            "  border-radius: 20px;"
            "  padding: 5px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #e0e0e0;"
            "}"
        )

    def open_file_dialog(self):
        try:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if not folder_path:
                return

            valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
            self.video_paths = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(valid_extensions)
            ]

            if not self.video_paths:
                QMessageBox.warning(self, "Warning", "선택한 폴더에 동영상 파일이 없습니다.")
                return

            self.current_video_index = 0
            self.run_button.setEnabled(True)
            self.set_roi_button.setEnabled(True)
            self.set_plate_roi_button.setEnabled(True)  # 번호판 ROI 버튼 활성화
            logger.info(f"선택된 폴더: {folder_path}")
            logger.info(
                f"처리할 동영상 파일 {len(self.video_paths)}개: {', '.join(os.path.basename(p) for p in self.video_paths[:5])}{'...' if len(self.video_paths) > 5 else ''}"
            )
            QMessageBox.information(self, "Info", f"{len(self.video_paths)}개의 동영상 파일이 선택되었습니다.")
        except Exception as e:
            logger.error(f"폴더 열기 중 오류: {str(e)}")
            QMessageBox.critical(self, "Error", f"폴더를 열 수 없습니다: {str(e)}")

    def open_single_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Video File",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
            if not file_path:
                return

            self.video_path = file_path
            self.video_paths = [self.video_path]
            self.current_video_index = 0
            self.run_button.setEnabled(True)
            self.set_roi_button.setEnabled(True)
            self.set_plate_roi_button.setEnabled(True)  # 번호판 ROI 버튼 활성화
            logger.info(f"선택된 파일: {self.video_path}")
            QMessageBox.information(self, "Info", f"선택된 파일: {os.path.basename(self.video_path)}")
        except Exception as e:
            logger.error(f"파일 열기 중 오류: {str(e)}")
            QMessageBox.critical(self, "Error", f"파일을 열 수 없습니다: {str(e)}")

    def start_drawing(self):
        try:
            if not hasattr(self, 'video_paths') or not self.video_paths:
                QMessageBox.warning(self, "Warning", "먼저 동영상을 포함한 폴더를 선택하세요!")
                return

            self.video_path = self.video_paths[0]
            logger.info(f"ROI 설정을 위한 동영상 경로: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", f"동영상 파일({os.path.basename(self.video_path)})을 열 수 없습니다.")
                logger.error(f"ROI 설정을 위한 동영상 파일을 열 수 없음: {self.video_path}")
                return

            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                QMessageBox.warning(self, "Error", "동영상에서 프레임을 읽을 수 없습니다.")
                logger.error("동영상 첫 프레임 읽기 실패")
                return

            orig_h, orig_w = frame.shape[:2]
            new_w, new_h = 1920, 1080
            logger.info(f"원본 해상도: {orig_w}x{orig_h}, 디스플레이 해상도: {new_w}x{new_h}")
            self.image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.temp_image = self.image.copy()
            self.scale_x = orig_w / new_w
            self.scale_y = orig_h / new_h
            logger.info(f"스케일 비율: x={self.scale_x:.4f}, y={self.scale_y:.4f}")
            cv2.namedWindow("Draw Rectangle", cv2.WINDOW_NORMAL)
            cv2.imshow("Draw Rectangle", self.temp_image)
            cv2.setMouseCallback("Draw Rectangle", self.mouse_draw_rectangle)
            cv2.waitKey(1)
            logger.info("ROI 설정 창 표시됨")
        except cv2.error as cv_err:
            logger.error(f"OpenCV 오류: {str(cv_err)}")
            QMessageBox.critical(self, "Error", f"동영상 처리 중 오류가 발생했습니다: {str(cv_err)}")
        except Exception as e:
            logger.error(f"ROI 설정 시작 중 오류: {str(e)}")
            QMessageBox.critical(self, "Error", f"ROI 설정을 시작할 수 없습니다: {str(e)}")

    def mouse_draw_rectangle(self, event, x, y, flags, param):
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.roi_x, self.roi_y = x, y
                self.temp_image = self.image.copy()
                logger.debug(f"ROI 그리기 시작: ({x}, {y})")
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.temp_image = self.image.copy()
                cv2.rectangle(self.temp_image, (self.roi_x, self.roi_y), (x, y), (0, 255, 0), 2)
                cv2.imshow("Draw Rectangle", self.temp_image)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.roi_width, self.roi_height = abs(x - self.roi_x), abs(y - self.roi_y)
                self.roi_x, self.roi_y = min(self.roi_x, x), min(self.roi_y, y)
                orig_x = int(self.roi_x * self.scale_x)
                orig_y = int(self.roi_y * self.scale_y)
                orig_w = int(self.roi_width * self.scale_x)
                orig_h = int(self.roi_height * self.scale_y)
                if orig_w < 10 or orig_h < 10:
                    QMessageBox.warning(self, "Warning", "ROI가 너무 작습니다. 다시 그려주세요.")
                    logger.warning(f"ROI가 너무 작음: {orig_w}x{orig_h} (최소 10x10)")
                    return
                self.roi_x, self.roi_y = orig_x, orig_y
                self.roi_width, self.roi_height = orig_w, orig_h
                cv2.rectangle(self.image,
                              (int(self.roi_x / self.scale_x), int(self.roi_y / self.scale_y)),
                              (int((self.roi_x + self.roi_width) / self.scale_x),
                               int((self.roi_y + self.roi_height) / self.scale_y)),
                              (0, 255, 0), 2)
                cv2.imshow("Draw Rectangle", self.image)
                cv2.waitKey(1)

                # ROI 유효성 검사 추가
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    # ROI 검증 및 재설정 요청
                    if self.validate_roi(width, height):
                        # ROI가 유효한 경우에만 저장
                        save_roi_settings(
                            self.roi_x,
                            self.roi_y,
                            self.roi_width,
                            self.roi_height,
                            self.config.min_size,
                            self.config.max_size
                        )
                        cv2.waitKey(500)
                        cv2.destroyWindow("Draw Rectangle")
                        QMessageBox.information(self, "ROI Set", "ROI가 성공적으로 설정되었습니다!")
                    else:
                        # ROI가 유효하지 않은 경우 창을 닫지 않고 유지
                        # 사용자가 다시 ROI를 설정할 수 있게 함
                        pass
                else:
                    # 비디오 파일을 열 수 없는 경우 기본 검증
                    if self.validate_roi():
                        save_roi_settings(
                            self.roi_x,
                            self.roi_y,
                            self.roi_width,
                            self.roi_height,
                            self.config.min_size,
                            self.config.max_size
                        )
                        cv2.waitKey(500)
                        cv2.destroyWindow("Draw Rectangle")
                        QMessageBox.information(self, "ROI Set", "ROI가 성공적으로 설정되었습니다!")

        except Exception as e:
            logger.error(f"ROI 설정(마우스 이벤트) 중 오류: {str(e)}")
            cv2.destroyAllWindows()
            QMessageBox.critical(self, "Error", f"ROI 설정 중 오류가 발생했습니다: {str(e)}")

    # 기존 DetectionApp 클래스의 start_detection 메서드를 수정
    def start_detection(self):
        try:
            # 비디오 파일의 크기 확인
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                # ROI 유효성 검사
                if not self.validate_roi(width, height):
                    # ROI가 유효하지 않으면 감지 시작하지 않음
                    return
            else:
                logger.error(f"비디오 파일을 열 수 없음: {self.video_path}")
                QMessageBox.critical(self, "오류", "비디오 파일을 열 수 없습니다!")
                return

            self.processing_enabled = True
            self.detection_running = True
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            os.makedirs(self.config.output_dir, exist_ok=True)
            logger.info("객체 검출 시작됨")

            # 쓰레기 투기 감지 모드
            try:
                from processing import VideoThread  # 기존 VideoThread 사용

                # 설정된 ROI 정보 확인
                logger.info(f"쓰레기 감지 시작 - ROI 정보: x={self.roi_x}, y={self.roi_y}, "
                            f"width={self.roi_width}, height={self.roi_height}")

                self.thread = VideoThread(
                    self.video_path,
                    self.config.min_size,
                    self.config.max_size,
                    100,  # min_box_size (고정값)
                    self.roi_x,
                    self.roi_y,
                    self.roi_width,
                    self.roi_height,
                    self.config
                )

                # 바로 번호판 인식으로 넘어가지 않도록 플래그 추가
                self.waited_for_detection = False

                # 쓰레기 감지 완료 시그널에 연결
                self.thread.littering_detected_signal.connect(self.on_littering_detected)

                # 비디오 저장 완료 시그널 연결
                self.thread.video_saved_signal.connect(self.process_saved_video)

                self.thread.change_pixmap_signal.connect(self.update_image)
                self.thread.finished.connect(self.on_detection_finished)
                self.thread.start()
                logger.info(f"VideoThread 시작: {os.path.basename(self.video_path)}")
            except Exception as e:
                logger.error(f"VideoThread 생성 및 시작 중 오류: {str(e)}")
                logger.error(traceback.format_exc())
                QMessageBox.critical(self, "Error", f"비디오 처리 중 오류가 발생했습니다: {str(e)}")
                self.run_button.setEnabled(True)
                self.stop_button.setEnabled(False)
        except Exception as e:
            logger.error(f"검출 시작 중 오류: {str(e)}")
            self.processing_enabled = False
            self.detection_running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            QMessageBox.critical(self, "Error", f"검출을 시작할 수 없습니다: {str(e)}")

    # 쓰레기 감지 이벤트 처리 핸들러 추가
    def on_littering_detected(self, bbox, class_name, confidence, strategy_results):
        """쓰레기 투기가 감지되었을 때 호출되는 콜백 함수"""
        logger.info(f"쓰레기 투기 감지됨: 클래스={class_name}, 신뢰도={confidence:.2f}")
        self.waited_for_detection = True  # 이 플래그로 감지가 발생했음을 기록

        # 추가 작업: 상태 표시, 소리 알림 등 가능
        try:
            # 상태바 업데이트 (상태바가 있는 경우)
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"쓰레기 투기 감지: 신뢰도 {confidence:.2f}")

            # 알림 소리 (선택적)
            # QApplication.beep()
        except Exception as e:
            logger.error(f"쓰레기 투기 감지 UI 업데이트 중 오류: {str(e)}")

    def process_saved_video(self, video_path):
        """
        저장된 비디오에서 번호판 인식 처리
        """
        # 쓰레기 투기 감지를 통해 영상이 저장되었는지 확인
        if not hasattr(self, 'waited_for_detection') or not self.waited_for_detection:
            logger.warning("쓰레기 투기 감지 없이 번호판 인식으로 넘어감. 감지 과정이 정상적으로 완료되지 않았을 수 있습니다.")

        self.saved_video_path = video_path
        logger.info(f"저장된 비디오 처리 시작: {video_path}")

        # 번호판 인식 활성화 여부 확인
        if not self.plate_recognition_checkbox.isChecked():
            logger.info("번호판 인식 체크박스가 비활성화되어 있어 번호판 인식을 건너뜁니다.")
            return

        # 영상이 저장되었으니 녹화 완료 메시지 표시 (UI 업데이트는 메인 스레드에서)
        QMessageBox.information(self, "녹화 완료", f"쓰레기 투기 영상 저장 완료\n파일: {os.path.basename(video_path)}")

        # 새 창에서 비디오 표시 및 번호판 인식 실행
        try:
            plate_dialog = PlateRecognitionDialog(
                video_path,
                self.plate_roi_x,
                self.plate_roi_y,
                self.plate_roi_width,
                self.plate_roi_height,
            )
            plate_dialog.exec_()  # 모달 다이얼로그로 실행
        except Exception as e:
            logger.error(f"번호판 인식 창 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.warning(self, "오류", f"번호판 인식 창 실행 중 오류가 발생했습니다: {str(e)}")

    def stop_detection(self):
        self.processing_enabled = False
        self.detection_running = False
        if self.thread:
            if hasattr(self.thread, 'stop'):
                self.thread.stop()
            else:
                self.thread._run_flag = False
                self.thread.wait()
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.current_video_index = 0
        print("Detection stopped completely.")

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def on_detection_finished(self):
        # PlateRecognitionThread 모드인 경우 추가 동영상 처리는 하지 않음
        if isinstance(self.thread, PlateRecognitionThread):
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            return

        # 기존 동영상 처리 모드라면 process_videos() 호출 (해당 메서드가 있다면)
        if hasattr(self, "process_videos"):
            self.process_videos()
        else:
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            # 처리 완료 팝업 추가
            self.show_completion_popup()

    def show_completion_popup(self):
        QMessageBox.information(self, "Info", "모든 동영상 파일 처리가 완료되었습니다!")
        print("모든 동영상 파일 처리가 완료되었습니다.")

    def start_plate_roi_drawing(self):
        try:
            if not hasattr(self, 'video_paths') or not self.video_paths:
                QMessageBox.warning(self, "Warning", "먼저 동영상을 포함한 폴더를 선택하세요!")
                return

            self.video_path = self.video_paths[0]
            logger.info(f"번호판 ROI 설정을 위한 동영상 경로: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", f"동영상 파일({os.path.basename(self.video_path)})을 열 수 없습니다.")
                logger.error(f"번호판 ROI 설정을 위한 동영상 파일을 열 수 없음: {self.video_path}")
                return

            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                QMessageBox.warning(self, "Error", "동영상에서 프레임을 읽을 수 없습니다.")
                logger.error("동영상 첫 프레임 읽기 실패")
                return

            orig_h, orig_w = frame.shape[:2]
            new_w, new_h = 1280, 720  # 적절한 크기로 조정
            logger.info(f"원본 해상도: {orig_w}x{orig_h}, 디스플레이 해상도: {new_w}x{new_h}")
            self.image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.temp_image = self.image.copy()
            self.scale_x = orig_w / new_w
            self.scale_y = orig_h / new_h
            logger.info(f"스케일 비율: x={self.scale_x:.4f}, y={self.scale_y:.4f}")

            # 모든 창 닫기 및 약간의 지연 추가
            cv2.destroyAllWindows()
            time.sleep(0.1)  # 창이 실제로 닫히도록 잠시 대기

            # 영문 이름으로 창 생성 (한글 인코딩 문제 방지)
            window_name = "Plate ROI Setting"
            self.plate_roi_window_name = window_name  # 인스턴스 변수로 저장

            # 창 생성 및 설정
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, new_w, new_h)
            cv2.imshow(window_name, self.temp_image)
            cv2.setMouseCallback(window_name, self.mouse_draw_plate_rectangle)

            # 창이 실제로 표시되고 포커스를 얻을 수 있도록 잠시 대기
            cv2.waitKey(100)

            # 그리기 모드 활성화
            self.drawing_plate_roi = True
            logger.info("번호판 ROI 설정 창 표시됨")
        except cv2.error as cv_err:
            logger.error(f"OpenCV 오류: {str(cv_err)}")
            QMessageBox.critical(self, "Error", f"동영상 처리 중 오류가 발생했습니다: {str(cv_err)}")
        except Exception as e:
            logger.error(f"번호판 ROI 설정 시작 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"번호판 ROI 설정을 시작할 수 없습니다: {str(e)}")

    def mouse_draw_plate_rectangle(self, event, x, y, flags, param):
        try:
            # 그리기 모드가 아니면 이벤트 처리 안함
            if not hasattr(self, 'drawing_plate_roi') or not self.drawing_plate_roi:
                return

            # 창 이름을 인스턴스 변수에서 가져옴
            window_name = self.plate_roi_window_name

            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.plate_roi_x, self.plate_roi_y = x, y
                self.temp_image = self.image.copy()
                logger.debug(f"번호판 ROI 그리기 시작: ({x}, {y})")

            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                # 임시 이미지에 사각형 그리기
                temp = self.image.copy()
                cv2.rectangle(temp, (self.plate_roi_x, self.plate_roi_y), (x, y), (0, 0, 255), 2)
                cv2.imshow(window_name, temp)

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.plate_roi_width, self.plate_roi_height = abs(x - self.plate_roi_x), abs(y - self.plate_roi_y)
                self.plate_roi_x, self.plate_roi_y = min(self.plate_roi_x, x), min(self.plate_roi_y, y)

                # 원본 좌표로 변환
                orig_x = int(self.plate_roi_x * self.scale_x)
                orig_y = int(self.plate_roi_y * self.scale_y)
                orig_w = int(self.plate_roi_width * self.scale_x)
                orig_h = int(self.plate_roi_height * self.scale_y)

                if orig_w < 10 or orig_h < 10:
                    QMessageBox.warning(self, "Warning", "번호판 ROI가 너무 작습니다. 다시 그려주세요.")
                    logger.warning(f"번호판 ROI가 너무 작음: {orig_w}x{orig_h} (최소 10x10)")
                    return

                # 최종 좌표 저장
                self.plate_roi_x, self.plate_roi_y = orig_x, orig_y
                self.plate_roi_width, self.plate_roi_height = orig_w, orig_h

                # 최종 ROI 표시
                result_image = self.image.copy()
                cv2.rectangle(result_image,
                              (int(self.plate_roi_x / self.scale_x), int(self.plate_roi_y / self.scale_y)),
                              (int((self.plate_roi_x + self.plate_roi_width) / self.scale_x),
                               int((self.plate_roi_y + self.plate_roi_height) / self.scale_y)),
                              (0, 0, 255), 2)
                cv2.imshow(window_name, result_image)

                # ROI 설정 저장
                self.save_plate_roi_settings(
                    self.plate_roi_x,
                    self.plate_roi_y,
                    self.plate_roi_width,
                    self.plate_roi_height
                )

                logger.info(
                    f"번호판 ROI(원본 좌표) 저장됨: x={self.plate_roi_x}, y={self.plate_roi_y}, "
                    f"width={self.plate_roi_width}, height={self.plate_roi_height}"
                )

                # 사용자에게 알림
                cv2.waitKey(1000)  # 잠시 대기하여 결과 확인할 수 있게 함
                cv2.destroyWindow(window_name)
                self.drawing_plate_roi = False  # 그리기 모드 종료

                # 약간의 지연 후 메시지 표시 (UI 스레드가 창 닫기를 처리할 시간 제공)
                QTimer.singleShot(100, lambda: QMessageBox.information(self, "번호판 ROI 설정", "번호판 ROI가 성공적으로 설정되었습니다!"))

        except Exception as e:
            logger.error(f"번호판 ROI 설정(마우스 이벤트) 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            cv2.destroyAllWindows()
            self.drawing_plate_roi = False
            QMessageBox.critical(self, "Error", f"번호판 ROI 설정 중 오류가 발생했습니다: {str(e)}")

    def save_plate_roi_settings(self, roi_x, roi_y, roi_width, roi_height):
        """번호판 ROI 설정을 JSON 파일로 저장"""
        try:
            roi_data = {
                "plate_roi_x": roi_x,
                "plate_roi_y": roi_y,
                "plate_roi_width": roi_width,
                "plate_roi_height": roi_height,
            }
            with open("plate_roi_settings.json", "w") as file:
                json.dump(roi_data, file, indent=4)
            logger.info("번호판 ROI 설정 저장 완료")
        except Exception as e:
            logger.error(f"번호판 ROI 설정 저장 실패: {str(e)}")

    def load_plate_roi_settings(self):
        """JSON 파일에서 번호판 ROI 설정 로드, 실패 시 기본값 반환"""
        logger.debug("load_plate_roi_settings 함수 호출됨.")
        try:
            with open("plate_roi_settings.json", "r") as file:
                roi_data = json.load(file)
                logger.info(f"번호판 ROI 데이터 로드 성공: {roi_data}")
                return (
                    max(0, roi_data.get("plate_roi_x", 0)),
                    max(0, roi_data.get("plate_roi_y", 300)),
                    max(1, roi_data.get("plate_roi_width", 400)),
                    max(1, roi_data.get("plate_roi_height", 150)),
                )
        except FileNotFoundError:
            logger.warning(f"번호판 ROI 설정 파일이 존재하지 않습니다. 기본값을 사용합니다.")
            return 0, 100, 400, 150
        except json.JSONDecodeError as e:
            logger.error(f"번호판 ROI 설정 파일 형식이 잘못되었습니다: {str(e)}")
            return 0, 100, 400, 150
        except Exception as e:
            logger.error(f"번호판 ROI 데이터 로드 실패: {str(e)}, 기본값 사용")
            return 0, 100, 400, 150

class PlateRecognitionDialog(QDialog):
    """
    저장된 비디오에서 번호판을 인식하는 다이얼로그
    """

    def __init__(self, video_path, roi_x, roi_y, roi_width, roi_height, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height

        # 자동 닫기 플래그와 타이머 초기화
        self.processing_completed = False
        self.close_timer = QTimer(self)
        self.close_timer.timeout.connect(self.close)
        self.close_timer.setSingleShot(True)

        self.setWindowTitle("번호판 인식")
        self.setGeometry(200, 200, 800, 600)

        # UI 초기화
        self.init_ui()

        # 번호판 인식 스레드 시작
        self.start_recognition()

    def init_ui(self):
        layout = QVBoxLayout()

        # 비디오 표시 영역
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(1280, 720)  # 더 큰 크기로 설정

        # 인식 결과 표시 영역
        result_group = QGroupBox("인식 결과")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("번호판 인식 중...")
        self.result_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 닫기 버튼
        self.close_button = QPushButton("닫기")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        self.setLayout(layout)
        # 좌표 정보 표시 라벨 추가
        self.coords_label = QLabel("좌표: 아직 인식되지 않음")
        self.coords_label.setStyleSheet("font-size: 12pt;")
        result_layout.addWidget(self.coords_label)

    def start_recognition(self):
        # 번호판 인식 스레드 시작
        self.plate_thread = PlateRecognitionThread(
            self.video_path,
            self.roi_x,
            self.roi_y,
            self.roi_width,
            self.roi_height,
            output_csv="plate_results.csv",
            frame_skip=2
        )

        # 결과 시그널 연결
        self.plate_thread.change_pixmap_signal.connect(self.update_image)
        self.plate_thread.plate_recognized_signal.connect(self.update_result)

        # 스레드 완료 시그널 연결 추가
        self.plate_thread.finished.connect(self.on_recognition_finished)

        # 스레드 시작
        self.plate_thread.start()
        self.log_message("번호판 인식 처리가 시작되었습니다.")

    def on_recognition_finished(self):
        """인식 스레드가 완료될 때 호출되는 함수"""
        self.processing_completed = True
        self.log_message("번호판 인식 처리가 완료되었습니다. 3초 후 창이 닫힙니다.")

        # 3초 후에 창 닫기
        self.close_timer.start(3000)  # 3000ms = 3초

    def update_image(self, cv_img):
        # 원본 해상도 유지하며 이미지 표시
        qt_img = self.convert_cv_qt(cv_img, keep_aspect=True)
        self.video_label.setPixmap(qt_img)

    def update_result(self, plate_text, confidence, coords):
        # 신뢰도 및 텍스트 업데이트
        self.result_label.setText(f"번호판: {plate_text} (신뢰도: {confidence}%)")

        # 좌표 정보 업데이트
        if coords and coords != (0, 0, 0, 0):
            x1, y1, x2, y2 = coords
            self.coords_label.setText(f"좌표: X1={x1}, Y1={y1}, X2={x2}, Y2={y2}")
            self.last_plate_coords = coords

        self.log_message(f"인식된 번호판: {plate_text} (신뢰도: {confidence}%)")

        # 인식 완료 시 스크린샷 저장 - 이미 스레드에서 처리하므로 여기서는 로그만
        if coords and coords != (0, 0, 0, 0):
            self.log_message(f"번호판 위치: X1={coords[0]}, Y1={coords[1]}, X2={coords[2]}, Y2={coords[3]}")
        # 최종 번호판 정보 업데이트
        if confidence > getattr(self, 'best_confidence', 0):
            self.best_confidence = confidence
            self.best_plate_text = plate_text
            self.best_coords = coords

            # UI 텍스트 갱신
            self.result_label.setText(f"최종 번호판: {plate_text} (신뢰도: {confidence}%)")

            # 좌표 정보도 갱신
            if coords and coords != (0, 0, 0, 0):
                x1, y1, x2, y2 = coords
                self.coords_label.setText(f"좌표: X1={x1}, Y1={y1}, X2={x2}, Y2={y2}")
                self.last_plate_coords = coords

            self.log_message(f"최종 번호판 업데이트: {plate_text} (신뢰도: {confidence}%)")

    def convert_cv_qt(self, cv_img, keep_aspect=True):
        """OpenCV 이미지를 Qt 이미지로 변환 - 원본 비율 유지 옵션 추가"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        if keep_aspect:
            # 원본 비율을 유지하면서 라벨 크기에 맞게 스케일링
            p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(),
                                            Qt.KeepAspectRatio)
        else:
            # 라벨 크기에 맞게 스트레칭 (비율 변경)
            p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height())

        return QPixmap.fromImage(p)

    # log_message 함수 추가
    def log_message(self, msg):
        """로그 메시지를 기록합니다."""
        print(msg)  # 간단한 콘솔 출력으로 대체
        # 더 복잡한 로깅 기능이 필요하면 여기에 추가