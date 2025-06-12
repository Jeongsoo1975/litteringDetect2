import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def put_text_with_korean(img, text, position, font_size=30, color=(0, 255, 0), thickness=2):
    """
    OpenCV 이미지에 한글 텍스트를 표시합니다.

    Args:
        img: OpenCV 이미지 (numpy 배열)
        text: 표시할 텍스트
        position: 텍스트 위치 (x, y)
        font_size: 폰트 크기
        color: 텍스트 색상, BGR 형식 (B, G, R)
        thickness: 텍스트 두께

    Returns:
        한글 텍스트가 추가된 이미지
    """
    # OpenCV BGR 이미지를 RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # PIL 이미지로 변환
    pil_img = Image.fromarray(img_rgb)

    # 이미지에 텍스트를 그리기 위한 드로우 객체 생성
    draw = ImageDraw.Draw(pil_img)

    # 폰트 로드 (시스템에 설치된 한글 폰트 사용)
    try:
        # 프로젝트 내부 폰트 확인 (우선순위 높음)
        font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
        project_fonts = [
            os.path.join(font_dir, "NanumGothic.ttf"),
            os.path.join(font_dir, "malgun.ttf")
        ]

        # 시스템 폰트 경로
        system_fonts = []
        if os.name == 'nt':  # Windows
            system_fonts = [
                "C:\\Windows\\Fonts\\malgun.ttf",  # 맑은 고딕
                "C:\\Windows\\Fonts\\gulim.ttc"  # 굴림
            ]
        else:  # Linux/Mac
            system_fonts = [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 한글이 없을 경우 대체 폰트
            ]

        # 모든 폰트 경로를 조합하여 존재하는 첫 번째 폰트 사용
        all_fonts = project_fonts + system_fonts
        for path in all_fonts:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        else:
            # 폰트를 찾지 못한 경우 기본 폰트 사용
            font = ImageFont.load_default()
            print("경고: 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다. 한글이 제대로 표시되지 않을 수 있습니다.")
    except Exception as e:
        # 폰트 로드 실패 시 기본 폰트 사용
        font = ImageFont.load_default()
        print(f"경고: 폰트 로드 중 오류 발생 - {str(e)}")

    # RGB 색상 순서를 BGR에서 변환
    color_rgb = (color[2], color[1], color[0])

    # 텍스트에 테두리 효과 추가 (가독성 향상)
    # 검은색 외곽선으로 텍스트 배경 그리기
    outline_offset = thickness
    for offset_x in range(-outline_offset, outline_offset + 1, outline_offset):
        for offset_y in range(-outline_offset, outline_offset + 1, outline_offset):
            if offset_x == 0 and offset_y == 0:
                continue
            draw.text((position[0] + offset_x, position[1] + offset_y), text, font=font, fill=(0, 0, 0))

    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color_rgb)

    # PIL 이미지를 다시 OpenCV 형식(BGR)으로 변환
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return result_img


def ensure_font_directory():
    """
    프로젝트에 fonts 디렉토리가 있는지 확인하고, 없으면 생성합니다.
    """
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
        print(f"fonts 디렉토리가 생성되었습니다: {font_dir}")
        print("이 디렉토리에 NanumGothic.ttf 또는 malgun.ttf 같은 한글 폰트를 복사해주세요.")
    return font_dir