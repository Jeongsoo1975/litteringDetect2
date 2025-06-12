# ROI 설정 개선 관련 핵심 수정 사항

class DetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LitteringDetection_v1.0")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.config = Config()
        self.current_video_index = 0
        self.processing_enabled = False
        
        # ROI 그리기 상태 관리 변수들 추가
        self.is_roi_drawing = False
        self.is_plate_roi_drawing = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.temp_roi_x = 0
        self.temp_roi_y = 0
        self.temp_roi_width = 0
        self.temp_roi_height = 0

        # ROI 설정값 불러오기 (기존 코드와 동일)
        try:
            self.roi_x, self.roi_y, self.roi_width, self.roi_height, self.min_size, self.max_size = load_roi_settings()
            logger.info(
                f"초기화된 ROI 값: x={self.roi_x}, y={self.roi_y}, width={self.roi_width}, height={self.roi_height}, "
                f"min_size={self.min_size}, max_size={self.max_size}"
            )
            self.config.min_size = self.min_size
            self.config.max_size = self.max_size
        except Exception as e:
            logger.error(f"ROI 설정 로드 실패: {str(e)}")
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = 0, 300, 800, 150
            QMessageBox.warning(self, "Warning", "ROI 설정을 불러오는데 실패했습니다. 기본값을 사용합니다.")

        # 번호판 ROI 설정값 불러오기 (기존 코드와 동일)
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

        # 상태 변수 초기화 (기존 코드와 동일)
        self.image = None
        self.temp_image = None
        self.video_path = None
        self.drawing = False
        self.detection_running = False
        self.thread = None
        self.saved_video_path = None
        self.plate_recognition_thread = None

        # UI 요소 생성
        self.init_ui()
        self.check_anpr_initialization()

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
            # 메인 스레드에서 안전하게 메시지박스 표시
            QTimer.singleShot(0, lambda: self.show_roi_validation_error(error_messages, frame_width, frame_height))
            return False

        return True

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
        
        # 메시지박스를 메인 스레드에서 안전하게 표시
        QMessageBox.warning(self, "ROI 설정 오류", full_message, QMessageBox.Ok)
        
        # ROI 설정 버튼을 강조하여 사용자의 주의를 끔
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
                if count < 6:  # 3번 깜빡임 (6번 상태 변경)
                    self.set_roi_button.setVisible(count % 2 == 0)
                    QApplication.processEvents()
                    QTimer.singleShot(300, lambda: blink_effect(count + 1))
                else:
                    # 원래 스타일로 복원
                    self.set_roi_button.setVisible(True)
                    self.set_roi_button.setStyleSheet(self._button_style())
            
            blink_effect()
            
        except Exception as e:
            logger.error(f"ROI 버튼 강조 중 오류: {str(e)}")

    def mouse_draw_rectangle(self, event, x, y, flags, param):
        """개선된 ROI 그리기 이벤트 핸들러 - 안전한 경계 검사"""
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.is_roi_drawing = True
                self.roi_start_x, self.roi_start_y = x, y
                self.temp_image = self.image.copy()
                logger.debug(f"ROI 그리기 시작: ({x}, {y})")
                
            elif event == cv2.EVENT_MOUSEMOVE and self.is_roi_drawing:
                # 실시간으로 ROI 미리보기 표시
                self.temp_image = self.image.copy()
                
                # 임시 ROI 좌표 계산
                self.temp_roi_x = min(self.roi_start_x, x)
                self.temp_roi_y = min(self.roi_start_y, y)
                self.temp_roi_width = abs(x - self.roi_start_x)
                self.temp_roi_height = abs(y - self.roi_start_y)
                
                # ROI 경계 검사를 통과하면 녹색, 실패하면 빨간색으로 표시
                display_w, display_h = self.temp_image.shape[1], self.temp_image.shape[0]
                
                # 원본 좌표로 변환
                orig_x = int(self.temp_roi_x * self.scale_x)
                orig_y = int(self.temp_roi_y * self.scale_y)
                orig_w = int(self.temp_roi_width * self.scale_x)
                orig_h = int(self.temp_roi_height * self.scale_y)
                
                # 실시간 검증
                is_valid_preview = (
                    orig_x >= 0 and orig_y >= 0 and
                    orig_x + orig_w <= self.video_frame_width and
                    orig_y + orig_h <= self.video_frame_height and
                    orig_w > 10 and orig_h > 10
                )
                
                # 색상 선택 (유효하면 녹색, 무효하면 빨간색)
                color = (0, 255, 0) if is_valid_preview else (0, 0, 255)
                
                cv2.rectangle(self.temp_image, (self.temp_roi_x, self.temp_roi_y), (x, y), color, 2)
                
                # 좌표 정보 표시
                coord_text = f"Size: {orig_w}x{orig_h}"
                cv2.putText(self.temp_image, coord_text, (self.temp_roi_x, self.temp_roi_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imshow("Draw Rectangle", self.temp_image)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_roi_drawing = False
                
                # 최종 ROI 좌표 계산
                self.temp_roi_width = abs(x - self.roi_start_x)
                self.temp_roi_height = abs(y - self.roi_start_y)
                self.temp_roi_x = min(self.roi_start_x, x)
                self.temp_roi_y = min(self.roi_start_y, y)
                
                # 원본 좌표로 변환
                orig_x = int(self.temp_roi_x * self.scale_x)
                orig_y = int(self.temp_roi_y * self.scale_y)
                orig_w = int(self.temp_roi_width * self.scale_x)
                orig_h = int(self.temp_roi_height * self.scale_y)
                
                # 최소 크기 검사
                if orig_w < 10 or orig_h < 10:
                    logger.warning(f"ROI가 너무 작음: {orig_w}x{orig_h} (최소 10x10)")
                    # 경고 메시지를 OpenCV 창에 표시
                    warning_image = self.image.copy()
                    cv2.putText(warning_image, "ROI too small! Please draw larger area.", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Draw Rectangle", warning_image)
                    cv2.waitKey(2000)  # 2초간 경고 표시
                    cv2.imshow("Draw Rectangle", self.image)  # 원본 이미지로 복원
                    return
                
                # 임시 ROI 값 설정
                self.roi_x, self.roi_y = orig_x, orig_y
                self.roi_width, self.roi_height = orig_w, orig_h
                
                # 비디오 프레임 크기 정보 가져오기
                if hasattr(self, 'video_frame_width') and hasattr(self, 'video_frame_height'):
                    frame_width = self.video_frame_width
                    frame_height = self.video_frame_height
                else:
                    # 기본값 사용
                    frame_width = 1920
                    frame_height = 1080
                
                # ROI 검증
                if self.validate_roi(frame_width, frame_height):
                    # 검증 통과 시 ROI 저장 및 창 닫기
                    cv2.rectangle(self.image,
                                  (int(self.roi_x / self.scale_x), int(self.roi_y / self.scale_y)),
                                  (int((self.roi_x + self.roi_width) / self.scale_x),
                                   int((self.roi_y + self.roi_height) / self.scale_y)),
                                  (0, 255, 0), 2)
                    cv2.imshow("Draw Rectangle", self.image)
                    cv2.waitKey(1000)  # 1초간 결과 표시
                    
                    # ROI 설정 저장
                    save_roi_settings(
                        self.roi_x, self.roi_y, self.roi_width, self.roi_height,
                        self.config.min_size, self.config.max_size
                    )
                    
                    cv2.destroyWindow("Draw Rectangle")
                    
                    # 성공 메시지는 메인 스레드에서 표시
                    QTimer.singleShot(100, lambda: QMessageBox.information(
                        self, "ROI 설정 완료", 
                        f"ROI가 성공적으로 설정되었습니다!\n위치: ({self.roi_x}, {self.roi_y})\n크기: {self.roi_width}x{self.roi_height}"
                    ))
                else:
                    # 검증 실패 시 다시 그리기 모드 유지
                    logger.warning("ROI 검증 실패 - 다시 설정해주세요")
                    # 실패한 ROI를 빨간색으로 표시
                    failed_image = self.image.copy()
                    cv2.rectangle(failed_image,
                                  (int(self.roi_x / self.scale_x), int(self.roi_y / self.scale_y)),
                                  (int((self.roi_x + self.roi_width) / self.scale_x),
                                   int((self.roi_y + self.roi_height) / self.scale_y)),
                                  (0, 0, 255), 2)
                    cv2.putText(failed_image, "Invalid ROI! Please redraw.", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Draw Rectangle", failed_image)
                    cv2.waitKey(2000)  # 2초간 오류 표시
                    cv2.imshow("Draw Rectangle", self.image)  # 원본으로 복원

        except Exception as e:
            logger.error(f"ROI 설정(마우스 이벤트) 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            cv2.destroyAllWindows()
            QMessageBox.critical(self, "Error", f"ROI 설정 중 오류가 발생했습니다: {str(e)}")

    def start_drawing(self):
        """개선된 ROI 설정 시작 함수"""
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
            if not ret or frame is None:
                cap.release()
                QMessageBox.warning(self, "Error", "동영상에서 프레임을 읽을 수 없습니다.")
                logger.error("동영상 첫 프레임 읽기 실패")
                return

            # 원본 프레임 크기 저장 (검증용)
            orig_h, orig_w = frame.shape[:2]
            self.video_frame_width = orig_w
            self.video_frame_height = orig_h
            
            # 디스플레이용 크기 조정
            new_w, new_h = 1920, 1080
            logger.info(f"원본 해상도: {orig_w}x{orig_h}, 디스플레이 해상도: {new_w}x{new_h}")
            
            self.image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.temp_image = self.image.copy()
            self.scale_x = orig_w / new_w
            self.scale_y = orig_h / new_h
            
            cap.release()
            
            logger.info(f"스케일 비율: x={self.scale_x:.4f}, y={self.scale_y:.4f}")
            
            # OpenCV 창 생성 및 설정
            cv2.destroyAllWindows()  # 기존 창 모두 닫기
            cv2.namedWindow("Draw Rectangle", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Draw Rectangle", new_w, new_h)
            cv2.imshow("Draw Rectangle", self.temp_image)
            cv2.setMouseCallback("Draw Rectangle", self.mouse_draw_rectangle)
            cv2.waitKey(1)
            
            logger.info("ROI 설정 창 표시됨")
            
        except cv2.error as cv_err:
            logger.error(f"OpenCV 오류: {str(cv_err)}")
            QMessageBox.critical(self, "Error", f"동영상 처리 중 오류가 발생했습니다: {str(cv_err)}")
        except Exception as e:
            logger.error(f"ROI 설정 시작 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"ROI 설정을 시작할 수 없습니다: {str(e)}")
