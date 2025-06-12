# 번호판 ROI 설정 개선 부분 (계속)

                    cv2.waitKey(2000)  # 2초간 경고 표시
                    cv2.imshow(window_name, self.image)  # 원본 이미지로 복원
                    return
                
                # 임시 ROI 값 설정
                self.plate_roi_x, self.plate_roi_y = orig_x, orig_y
                self.plate_roi_width, self.plate_roi_height = orig_w, orig_h
                
                # 비디오 프레임 크기 정보 가져오기
                if hasattr(self, 'video_frame_width') and hasattr(self, 'video_frame_height'):
                    frame_width = self.video_frame_width
                    frame_height = self.video_frame_height
                else:
                    frame_width = 1920
                    frame_height = 1080
                
                # 번호판 ROI 검증
                if self.validate_plate_roi(frame_width, frame_height):
                    # 검증 통과 시 ROI 저장 및 창 닫기
                    result_image = self.image.copy()
                    cv2.rectangle(result_image,
                                  (int(self.plate_roi_x / self.scale_x), int(self.plate_roi_y / self.scale_y)),
                                  (int((self.plate_roi_x + self.plate_roi_width) / self.scale_x),
                                   int((self.plate_roi_y + self.plate_roi_height) / self.scale_y)),
                                  (0, 0, 255), 2)
                    cv2.imshow(window_name, result_image)
                    cv2.waitKey(1000)  # 1초간 결과 표시
                    
                    # 번호판 ROI 설정 저장
                    self.save_plate_roi_settings(
                        self.plate_roi_x, self.plate_roi_y, 
                        self.plate_roi_width, self.plate_roi_height
                    )
                    
                    cv2.destroyWindow(window_name)
                    self.drawing_plate_roi = False
                    
                    logger.info(
                        f"번호판 ROI(원본 좌표) 저장됨: x={self.plate_roi_x}, y={self.plate_roi_y}, "
                        f"width={self.plate_roi_width}, height={self.plate_roi_height}"
                    )
                    
                    # 성공 메시지는 메인 스레드에서 표시
                    QTimer.singleShot(100, lambda: QMessageBox.information(
                        self, "번호판 ROI 설정 완료",
                        f"번호판 ROI가 성공적으로 설정되었습니다!\n위치: ({self.plate_roi_x}, {self.plate_roi_y})\n크기: {self.plate_roi_width}x{self.plate_roi_height}"
                    ))
                else:
                    # 검증 실패 시 다시 그리기 모드 유지
                    logger.warning("번호판 ROI 검증 실패 - 다시 설정해주세요")
                    failed_image = self.image.copy()
                    cv2.rectangle(failed_image,
                                  (int(self.plate_roi_x / self.scale_x), int(self.plate_roi_y / self.scale_y)),
                                  (int((self.plate_roi_x + self.plate_roi_width) / self.scale_x),
                                   int((self.plate_roi_y + self.plate_roi_height) / self.scale_y)),
                                  (0, 0, 139), 2)  # 진한 빨간색
                    cv2.putText(failed_image, "Invalid Plate ROI! Please redraw.", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)
                    cv2.imshow(window_name, failed_image)
                    cv2.waitKey(2000)  # 2초간 오류 표시
                    cv2.imshow(window_name, self.image)  # 원본으로 복원

        except Exception as e:
            logger.error(f"번호판 ROI 설정(마우스 이벤트) 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            cv2.destroyAllWindows()
            self.drawing_plate_roi = False
            QMessageBox.critical(self, "Error", f"번호판 ROI 설정 중 오류가 발생했습니다: {str(e)}")

    def start_plate_roi_drawing(self):
        """개선된 번호판 ROI 설정 시작 함수"""
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
            if not ret or frame is None:
                cap.release()
                QMessageBox.warning(self, "Error", "동영상에서 프레임을 읽을 수 없습니다.")
                logger.error("동영상 첫 프레임 읽기 실패")
                return

            # 원본 프레임 크기 저장
            orig_h, orig_w = frame.shape[:2]
            self.video_frame_width = orig_w
            self.video_frame_height = orig_h
            
            # 디스플레이용 크기 조정
            new_w, new_h = 1280, 720  # 번호판 ROI는 조금 작은 창으로
            logger.info(f"원본 해상도: {orig_w}x{orig_h}, 디스플레이 해상도: {new_w}x{new_h}")
            
            self.image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.temp_image = self.image.copy()
            self.scale_x = orig_w / new_w
            self.scale_y = orig_h / new_h
            
            cap.release()
            
            logger.info(f"스케일 비율: x={self.scale_x:.4f}, y={self.scale_y:.4f}")

            # 모든 창 닫기 및 약간의 지연 추가
            cv2.destroyAllWindows()
            time.sleep(0.1)

            # 영문 이름으로 창 생성 (한글 인코딩 문제 방지)
            window_name = "Plate ROI Setting"
            self.plate_roi_window_name = window_name

            # 창 생성 및 설정
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, new_w, new_h)
            cv2.imshow(window_name, self.temp_image)
            cv2.setMouseCallback(window_name, self.mouse_draw_plate_rectangle)

            # 창이 실제로 표시되고 포커스를 얻을 수 있도록 잠시 대기
            cv2.waitKey(100)

            # 그리기 모드 활성화
            self.drawing_plate_roi = True
            self.is_plate_roi_drawing = False  # 마우스 드래그 상태 초기화
            logger.info("번호판 ROI 설정 창 표시됨")
            
        except cv2.error as cv_err:
            logger.error(f"OpenCV 오류: {str(cv_err)}")
            QMessageBox.critical(self, "Error", f"동영상 처리 중 오류가 발생했습니다: {str(cv_err)}")
        except Exception as e:
            logger.error(f"번호판 ROI 설정 시작 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"번호판 ROI 설정을 시작할 수 없습니다: {str(e)}")

    def start_detection(self):
        """개선된 감지 시작 함수 - ROI 검증 강화"""
        try:
            # 현재 비디오 경로 확인
            if not hasattr(self, 'video_paths') or not self.video_paths:
                QMessageBox.warning(self, "Warning", "먼저 동영상 파일을 선택하세요!")
                return
                
            current_video = self.video_paths[self.current_video_index] if hasattr(self, 'current_video_index') else self.video_paths[0]
            
            # 비디오 파일의 크기 확인
            cap = cv2.VideoCapture(current_video)
            if not cap.isOpened():
                logger.error(f"비디오 파일을 열 수 없음: {current_video}")
                QMessageBox.critical(self, "오류", "비디오 파일을 열 수 없습니다!")
                return
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            logger.info(f"감지 시작 - 비디오 해상도: {width}x{height}")

            # ROI 유효성 검사
            if not self.validate_roi(width, height):
                logger.warning("쓰레기 ROI 검증 실패로 인해 감지를 시작할 수 없습니다.")
                return
                
            # 번호판 인식이 활성화된 경우 번호판 ROI도 검증
            if self.plate_recognition_checkbox.isChecked():
                if not self.validate_plate_roi(width, height):
                    logger.warning("번호판 ROI 검증 실패로 인해 감지를 시작할 수 없습니다.")
                    return

            # 모든 검증을 통과한 경우에만 감지 시작
            self.processing_enabled = True
            self.detection_running = True
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            os.makedirs(self.config.output_dir, exist_ok=True)
            logger.info("객체 검출 시작됨")

            # VideoThread 시작 (기존 코드와 동일)
            try:
                from processing import VideoThread

                logger.info(f"쓰레기 감지 시작 - ROI 정보: x={self.roi_x}, y={self.roi_y}, "
                            f"width={self.roi_width}, height={self.roi_height}")

                self.thread = VideoThread(
                    current_video,
                    self.config.min_size,
                    self.config.max_size,
                    100,  # min_box_size (고정값)
                    self.roi_x,
                    self.roi_y,
                    self.roi_width,
                    self.roi_height,
                    self.config
                )

                self.waited_for_detection = False

                # 시그널 연결
                self.thread.littering_detected_signal.connect(self.on_littering_detected)
                self.thread.video_saved_signal.connect(self.process_saved_video)
                self.thread.change_pixmap_signal.connect(self.update_image)
                self.thread.finished.connect(self.on_detection_finished)
                self.thread.start()
                
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
