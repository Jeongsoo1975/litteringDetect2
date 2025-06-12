# anpr_api.py

from ctypes import cdll, c_char_p, c_int32
import platform
import os
import sys
import logging


def getLibPath():
    os_name = platform.system().lower()
    arch_name = platform.machine().lower()
    if os_name == 'windows':
        if arch_name in ['x86_64', 'amd64']:
            return os.path.join('bin', 'windows-x86_64', 'tsanpr.dll')
        elif arch_name == 'x86':
            return os.path.join('bin', 'windows-x86', 'tsanpr.dll')
    elif os_name == 'linux':
        if arch_name == 'x86_64':
            return os.path.join('bin', 'linux-x86_64', 'libtsanpr.so')
        elif arch_name == 'aarch64':
            return os.path.join('bin', 'linux-aarch64', 'libtsanpr.so')
    print('Unsupported target platform')
    sys.exit(-1)

# 라이브러리 경로 설정
LIB_PATH = getLibPath()
logging.info(f"ANPR 라이브러리 경로: {LIB_PATH}")

try:
    lib = cdll.LoadLibrary(LIB_PATH)
    # TS-ANPR API 함수 설정
    lib.anpr_initialize.argtype = c_char_p
    lib.anpr_initialize.restype = c_char_p

    lib.anpr_read_pixels.argtypes = (c_char_p, c_int32, c_int32, c_int32, c_char_p, c_char_p, c_char_p)
    lib.anpr_read_pixels.restype = c_char_p
except Exception as e:
    logging.error(f"라이브러리 로드 실패: {str(e)}")
    lib = None
    

def initialize():
    if lib is None:
        logging.error("라이브러리가 로드되지 않았습니다.")
        return "라이브러리 로드 실패"
        
    error = lib.anpr_initialize('text'.encode('utf-8'))
    result = error.decode('utf8') if error else error
    logging.info(f"TS-ANPR 라이브러리 초기화 결과: {result if result else '성공'}")
    return result


def getPixelFormat(shape, dtype):
    """
    이미지 형태와 데이터 타입을 기반으로 픽셀 형식 결정
    """
    if len(shape) == 2:
        return 'GRAY'
    elif len(shape) == 3:
        channels = shape[2]
        if channels == 3:
            return 'RGB'
        elif channels == 4:
            return 'RGBA'
    return 'RGB'  # 기본값


def readPixelsFromArray(image_array, outputFormat, options):
    """
    이미지 배열에서 번호판을 인식하는 함수

    Args:
        image_array: numpy array (이미지 데이터)
        outputFormat: 출력 데이터 형식 ('json', 'text' 등)
        options: 기능 옵션 ('v', 'vm', 'vs' 등)

    Returns:
        문자열: 인식 결과 (UTF-8 인코딩)
    """
    try:
        if lib is None:
            logging.error("라이브러리가 로드되지 않았습니다.")
            return ""
            
        height, width = image_array.shape[:2]

        # 이미지 픽셀 형식 판단
        pixelFormat = getPixelFormat(image_array.shape, image_array.dtype)

        # 로깅 제거 (디버깅 출력 감소)

        # numpy array를 바이트로 변환
        pixel_bytes = image_array.tobytes()

        # 인식 함수 호출
        result = lib.anpr_read_pixels(
            pixel_bytes, width, height, 0,
            pixelFormat.encode('utf-8'),
            outputFormat.encode('utf-8'),
            options.encode('utf-8')
        )

        decoded_result = result.decode('utf8')

        # 로깅 제거 (디버깅 출력 감소)

        return decoded_result

    except Exception as e:
        # 로깅 간소화
        logging.error(f"번호판 인식 API 오류: {str(e)}")
        return ""