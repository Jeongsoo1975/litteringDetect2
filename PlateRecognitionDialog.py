#PlateRecognitionDialog.py

import os
import cv2
import json
import csv
import time
import logging
import traceback
from datetime import datetime
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image, ImageDraw, ImageFont

# 한글 텍스트 처리 유틸리티 임포트
from korean_text_utils import put_text_with_korean

# TS-ANPR API 모듈 import (예외 처리 추가)
try:
    import anpr_api
    ANPR_AVAILABLE = True
    logging.info("TS-ANPR API 모듈 로드 성공")
except Exception as e:
    ANPR_AVAILABLE = False
    logging.error(f"TS-ANPR API 모듈 로드 실패: {str(e)}")


class PlateRecognitionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    plate_recognized_signal = pyqtSignal(str, int, tuple)  # 번호판, 신뢰도, 좌표

    def __init__(self, video_path, roi_x, roi_y, roi_width, roi_height, output_csv="plate_results.csv", frame_skip=2,
                 parent=None):
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
                    logging.error("TS-ANPR 초기화 오류: " + self.api_error.replace("‑", "-"))
                else:
                    logging.info("TS-ANPR 초기화 성공")
            except Exception as e:
                self.api_error = str(e)
                logging.error(f"TS-ANPR 초기화 중 예외 발생: {str(e)}")
        else:
            self.api_error = "TS-ANPR 모듈이 로드되지 않았습니다."
            logging.warning("TS-ANPR 모듈이 로드되지 않아 번호판 인식 기능이 제한됩니다.")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"번호판인식 스레드: 비디오 열기 실패 - {self.video_path}")
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
            plate_coords = None  # 번호판 좌표 정보

            for size_name, img in resized_versions:
                if recognized and best_ocr_conf > 0.7:  # 충분히 높은 신뢰도면 더 시도하지 않음
                    break

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

                                    # 좌표 정보 가져오기
                                    x_pos = area.get("x", 0)
                                    y_pos = area.get("y", 0)
                                    width = area.get("width", 0)
                                    height = area.get("height", 0)

                                    # ROI 상대 좌표를 프레임 절대 좌표로 변환
                                    abs_x = x_pos + self.roi_x
                                    abs_y = y_pos + self.roi_y

                                    if plate_text and ocr_conf > best_ocr_conf:
                                        best_plate_text = plate_text
                                        best_ocr_conf = ocr_conf
                                        best_plate_conf = plate_conf
                                        plate_x_position = x_pos
                                        recognized = True
                                        # 절대 좌표로 바운딩 박스 설정
                                        plate_coords = (
                                            abs_x, abs_y,
                                            abs_x + width, abs_y + height
                                        )

                                elif isinstance(item, dict) and "licensePlate" in item:
                                    # licensePlate 필드가 있는 경우 (객체 인식 결과)
                                    for plate in item.get("licensePlate", []):
                                        plate_text = plate.get("text", "")
                                        ocr_conf = plate.get("conf", {}).get("ocr", 0)
                                        plate_conf = plate.get("conf", {}).get("plate", 0)
                                        area = plate.get("area", {})

                                        # 좌표 정보 가져오기
                                        x_pos = area.get("x", 0)
                                        y_pos = area.get("y", 0)
                                        width = area.get("width", 0)
                                        height = area.get("height", 0)

                                        # ROI 상대 좌표를 프레임 절대 좌표로 변환
                                        abs_x = x_pos + self.roi_x
                                        abs_y = y_pos + self.roi_y

                                        if plate_text and ocr_conf > best_ocr_conf:
                                            best_plate_text = plate_text
                                            best_ocr_conf = ocr_conf
                                            best_plate_conf = plate_conf
                                            plate_x_position = x_pos
                                            recognized = True
                                            # 절대 좌표로 바운딩 박스 설정
                                            plate_coords = (
                                                abs_x, abs_y,
                                                abs_x + width, abs_y + height
                                            )

                        elif isinstance(result_data, dict):
                            # 오류 또는 직접 객체인 경우
                            if "licensePlate" in result_data:
                                # 직접 객체인 경우 처리
                                for plate in result_data.get("licensePlate", []):
                                    plate_text = plate.get("text", "")
                                    ocr_conf = plate.get("conf", {}).get("ocr", 0)
                                    plate_conf = plate.get("conf", {}).get("plate", 0)
                                    area = plate.get("area", {})

                                    # 좌표 정보 가져오기
                                    x_pos = area.get("x", 0)
                                    y_pos = area.get("y", 0)
                                    width = area.get("width", 0)
                                    height = area.get("height", 0)

                                    # ROI 상대 좌표를 프레임 절대 좌표로 변환
                                    abs_x = x_pos + self.roi_x
                                    abs_y = y_pos + self.roi_y

                                    if plate_text and ocr_conf > best_ocr_conf:
                                        best_plate_text = plate_text
                                        best_ocr_conf = ocr_conf
                                        best_plate_conf = plate_conf
                                        plate_x_position = x_pos
                                        recognized = True
                                        # 절대 좌표로 바운딩 박스 설정
                                        plate_coords = (
                                            abs_x, abs_y,
                                            abs_x + width, abs_y + height
                                        )
                    except Exception as e:
                        logging.debug(f"JSON 파싱 오류: {str(e)}")
                        pass

                except Exception as e:
                    logging.debug(f"이미지 처리 오류: {str(e)}")
                    pass

            # 인식 결과가 있는 경우 처리
            if recognized and best_plate_text:
                # 좌표 정보가 유효한지 확인
                if plate_coords:
                    x1, y1, x2, y2 = plate_coords
                    self.last_plate_coords = plate_coords

                    # 번호판 바운딩 박스 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 녹색으로 번호판 표시

                    # PIL을 사용한 한글 텍스트 표시
                    frame = put_text_with_korean(
                        frame,
                        best_plate_text,
                        (x1, y1 - 30),  # 텍스트 위치 조정
                        font_size=24,
                        color=(0, 255, 0)  # 녹색
                    )

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
                    # 인식 결과 시그널 발생
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
                    # 최종 번호판 시그널 발생
                    self.plate_recognized_signal.emit(final_plate_text, confidence_percentage,
                                                      self.last_plate_coords if self.last_plate_coords else (
                                                      0, 0, 0, 0))

            # ROI 영역 표시 (항상 빨간색으로 표시)
            cv2.rectangle(frame, (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                          (0, 0, 255), 2)  # 빨간색 박스

            self.change_pixmap_signal.emit(frame)
            time.sleep(0.03)  # UI 업데이트 시간 여유

        cap.release()

        # 비디오 처리 완료 후, 최종 번호판 저장
        if best_overall_image is not None and best_overall_plate_text:
            # 최종 결과 이미지에 바운딩 박스 표시
            if best_overall_coords:
                x1, y1, x2, y2 = best_overall_coords
                cv2.rectangle(best_overall_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 바운딩 박스

                # 이미지 상단에 최종 번호판 텍스트 표시 (한글 지원)
                best_overall_image = put_text_with_korean(
                    best_overall_image,
                    f"최종 번호판: {best_overall_plate_text} (신뢰도: {best_overall_confidence}%)",
                    (10, 30),
                    font_size=30,
                    color=(0, 255, 0)
                )

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

        self.setWindowTitle("번호판 인식")
        self.setGeometry(200, 200, 1000, 800)  # 창 크기를 더 크게 설정

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
        self.video_label.setMinimumSize(800, 600)  # 비디오 영역 크기 설정
        layout.addWidget(self.video_label)

        # 인식 결과 표시 영역
        result_group = QGroupBox("인식 결과")
        result_layout = QVBoxLayout()

        self.result_label = QLabel("번호판 인식 중...")
        self.result_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        result_layout.addWidget(self.result_label)

        # 디버깅 로그 영역 추가
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        self.log_text.setMinimumHeight(150)
        result_layout.addWidget(self.log_text)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 컨트롤 버튼 영역
        buttons_layout = QHBoxLayout()

        # 닫기 버튼
        self.close_button = QPushButton("닫기")
        self.close_button.setMinimumHeight(40)
        self.close_button.setStyleSheet("font-weight: bold;")
        self.close_button.clicked.connect(self.close)

        # 저장 버튼
        self.save_button = QPushButton("결과 저장")
        self.save_button.setMinimumHeight(40)
        self.save_button.setStyleSheet("font-weight: bold;")
        self.save_button.clicked.connect(self.save_results)

        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.close_button)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

        # 로고 및 초기 메시지 표시
        self.log_message("번호판 인식 시작중...")
        self.log_message(f"동영상 파일: {os.path.basename(self.video_path)}")
        self.log_message(f"번호판 ROI: X={self.roi_x}, Y={self.roi_y}, 너비={self.roi_width}, 높이={self.roi_height}")
        self.log_message("=" * 50)

    def log_message(self, msg):
        """로그 메시지를 디버깅 영역에 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        # 스크롤을 항상 최하단으로
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

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

        # 스레드 시작
        self.plate_thread.start()
        self.log_message("번호판 인식 처리가 시작되었습니다.")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_result(self, plate_text, confidence):
        self.result_label.setText(f"번호판: {plate_text} (신뢰도: {confidence}%)")
        self.log_message(f"인식된 번호판: {plate_text} (신뢰도: {confidence}%)")

    def convert_cv_qt(self, cv_img):
        """OpenCV 이미지를 Qt 이미지로 변환"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def save_results(self):
        """인식 결과를 별도 파일로 저장"""
        try:
            filename = f"plate_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"번호판 인식 결과 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"비디오 파일: {self.video_path}\n")
                f.write(f"ROI 좌표: X={self.roi_x}, Y={self.roi_y}, 너비={self.roi_width}, 높이={self.roi_height}\n")
                f.write("=" * 50 + "\n")
                f.write(self.log_text.toPlainText())

            self.log_message(f"결과가 '{filename}' 파일에 저장되었습니다.")
            QMessageBox.information(self, "저장 완료", f"인식 결과가 '{filename}' 파일에 저장되었습니다.")
        except Exception as e:
            self.log_message(f"결과 저장 중 오류 발생: {str(e)}")
            QMessageBox.warning(self, "저장 오류", f"결과 저장 중 오류가 발생했습니다: {str(e)}")

    def closeEvent(self, event):
        # 스레드 종료
        if hasattr(self, 'plate_thread') and self.plate_thread.isRunning():
            self.plate_thread._run_flag = False
            self.plate_thread.wait()
        event.accept()