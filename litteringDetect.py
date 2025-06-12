#litteringDetect.py
import sys
import os
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ui import DetectionApp
from ultralytics import YOLO
from processing import logger


def load_model():
    """
    YOLO 모델 로드 함수
    - 실행 파일이 있는 폴더(혹은 스크립트 폴더)에 yolov8n.pt가 존재해야 함.
    """
    try:
        # 실행 파일(.exe) 혹은 현재 파이썬 스크립트가 위치한 디렉토리
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        # 모델 파일명
        model_filename = "yolov8n.pt"

        # 같은 폴더 내 모델 경로
        model_path = os.path.join(base_dir, model_filename)

        # 모델 파일이 없으면 예외
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일 '{model_filename}' 이(가) 존재하지 않습니다. "
                f"실행 파일과 같은 폴더 내에 '{model_filename}' 를 위치시켜주세요."
            )

        logger.info(f"모델 로드 시작: {model_path}")
        model = YOLO(model_path)
        logger.info("모델 로드 완료")
        return model

    except FileNotFoundError as e:
        logger.error(f"모델 파일을 찾을 수 없음: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"모델 로드 중 예외 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """메인 함수"""
    try:
        # Qt 애플리케이션 초기화
        app = QApplication(sys.argv)

        # Step 1: 로딩 팝업(QSplashScreen) 생성 및 표시
        splash_pix = QPixmap(400, 200)  # 팝업 크기 설정
        splash_pix.fill(Qt.white)  # 배경 흰색 설정
        splash = QSplashScreen(splash_pix)
        splash.showMessage(
            "프로그램 로딩 중...\n잠시만 기다려주세요.",
            Qt.AlignCenter | Qt.AlignBottom,
            Qt.black
        )
        splash.show()
        app.processEvents()  # 팝업을 즉시 표시

        # Step 2: 초기화 작업 (모델 로드)
        try:
            # 가중치 파일을 로드
            model = load_model()
        except FileNotFoundError as e:
            # 오류 발생 시 메시지 박스 표시 후 종료
            QMessageBox.critical(None, "오류", str(e))
            splash.close()
            sys.exit(1)
        except Exception as e:
            QMessageBox.critical(None, "오류", f"모델 로드 중 오류가 발생했습니다: {str(e)}")
            splash.close()
            sys.exit(1)

        # Step 3: DetectionApp 메인 윈도우 생성 및 표시
        try:
            window = DetectionApp()
            splash.finish(window)  # 팝업 종료 후 메인 윈도우로 전환
            window.show()
        except Exception as e:
            logger.error(f"UI 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(None, "오류", f"애플리케이션 초기화 중 오류가 발생했습니다: {str(e)}")
            splash.close()
            sys.exit(1)

        # Step 4: 이벤트 루프 실행
        sys.exit(app.exec_())

    except Exception as e:
        logger.error(f"애플리케이션 실행 중 예외 발생: {str(e)}")
        logger.error(traceback.format_exc())
        QMessageBox.critical(None, "오류", f"예기치 않은 오류가 발생했습니다: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()