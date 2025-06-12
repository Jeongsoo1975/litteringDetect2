                logger.info(f"VideoThread 시작: {os.path.basename(current_video)}")
                
            except Exception as e:
                logger.error(f"VideoThread 생성 및 시작 중 오류: {str(e)}")
                logger.error(traceback.format_exc())
                QMessageBox.critical(self, "Error", f"비디오 처리 중 오류가 발생했습니다: {str(e)}")
                self.run_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                
        except Exception as e:
            logger.error(f"검출 시작 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            self.processing_enabled = False
            self.detection_running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            QMessageBox.critical(self, "Error", f"검출을 시작할 수 없습니다: {str(e)}")

    def on_littering_detected(self, bbox, class_name, confidence, strategy_results):
        """쓰레기 투기가 감지되었을 때 호출되는 콜백 함수"""
        logger.info(f"쓰레기 투기 감지됨: 클래스={class_name}, 신뢰도={confidence:.2f}")
        self.waited_for_detection = True

        try:
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"쓰레기 투기 감지: 신뢰도 {confidence:.2f}")
        except Exception as e:
            logger.error(f"쓰레기 투기 감지 UI 업데이트 중 오류: {str(e)}")

    def process_saved_video(self, video_path):
        """저장된 비디오에서 번호판 인식 처리"""
        if not hasattr(self, 'waited_for_detection') or not self.waited_for_detection:
            logger.warning("쓰레기 투기 감지 없이 번호판 인식으로 넘어감. 감지 과정이 정상적으로 완료되지 않았을 수 있습니다.")

        self.saved_video_path = video_path
        logger.info(f"저장된 비디오 처리 시작: {video_path}")

        # 번호판 인식 활성화 여부 확인
        if not self.plate_recognition_checkbox.isChecked():
            logger.info("번호판 인식 체크박스가 비활성화되어 있어 번호판 인식을 건너뜁니다.")
            return

        # 영상이 저장되었으니 녹화 완료 메시지 표시
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
            plate_dialog.exec_()
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
        if isinstance(self.thread, PlateRecognitionThread):
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            return

        if hasattr(self, "process_videos"):
            self.process_videos()
        else:
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.show_completion_popup()

    def show_completion_popup(self):
        QMessageBox.information(self, "Info", "모든 동영상 파일 처리가 완료되었습니다!")
        print("모든 동영상 파일 처리가 완료되었습니다.")

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
    """저장된 비디오에서 번호판을 인식하는 다이얼로그"""

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
        self.video_label.setMinimumSize(1280, 720)

        # 인식 결과 표시 영역
        result_group = QGroupBox("인식 결과")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("번호판 인식 중...")
        self.result_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        result_layout.addWidget(self.result_label)

        # 좌표 정보 표시 라벨 추가
        self.coords_label = QLabel("좌표: 아직 인식되지 않음")
        self.coords_label.setStyleSheet("font-size: 12pt;")
        result_layout.addWidget(self.coords_label)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 닫기 버튼
        self.close_button = QPushButton("닫기")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

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
        self.close_timer.start(3000)

    def update_image(self, cv_img):
        # 원본 해상도 유지하며 이미지 표시
        qt_img = self.convert_cv_qt(cv_img, keep_aspect=True)
        self.video_label.setPixmap(qt_img)

    def update_result(self, plate_text, confidence, coords):
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

    def log_message(self, msg):
        """로그 메시지를 기록합니다."""
        print(msg)


# 메인 실행 부분
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = DetectionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 치명적 오류: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"애플리케이션을 시작할 수 없습니다: {str(e)}")
