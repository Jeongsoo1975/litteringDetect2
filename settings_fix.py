# settings_fix.py - 타임아웃 문제 해결을 위한 개선된 설정 파일 로딩 함수

import os
import logging
import time
import threading
from pathlib import Path

def load_settings_from_file_safe(file_path="default_settings.txt", timeout=5):
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
            print(f"경고: 설정 파일이 없습니다: {file_path}. 기본값을 사용합니다.")
            return {}
        
        # 파일 크기 확인 (너무 크면 문제가 있을 수 있음)
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024:  # 10KB 초과
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
                    print(f"경고: 설정 파일이 비어있습니다: {file_path}")
                    return
                
                # 줄 단위로 파싱 (최대 100줄까지만 처리)
                lines = content.split('\n')
                if len(lines) > 100:
                    print(f"경고: 설정 파일의 줄 수가 너무 많습니다: {len(lines)}줄. 최대 100줄까지만 처리됩니다.")
                    lines = lines[:100]
                
                parsed_settings = {}
                line_count = 0
                
                for line in lines:
                    line_count += 1
                    if line_count % 10 == 0:  # 10줄마다 처리 중임을 표시
                        print(f"설정 파일 처리 중... {line_count}/{len(lines)}줄")
                    
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
                            print(f"경고: 설정 파일의 {line_count}줄에서 값 변환 오류: {ve}")
                            continue
                    else:
                        print(f"경고: 설정 파일의 {line_count}줄이 올바르지 않은 형식입니다: {line}")
                
                result = parsed_settings
                
            except UnicodeDecodeError:
                print(f"오류: 설정 파일 인코딩 문제: {file_path}. UTF-8로 저장해주세요.")
                exception_occurred.set()
            except PermissionError:
                print(f"오류: 설정 파일 접근 권한 없음: {file_path}")
                exception_occurred.set()
            except Exception as e:
                print(f"오류: 설정 파일 읽기 중 예외 발생: {str(e)}")
                exception_occurred.set()
        
        # 별도 스레드에서 파일 읽기 실행
        read_thread = threading.Thread(target=read_file, daemon=True)
        read_thread.start()
        read_thread.join(timeout=timeout)
        
        # 타임아웃 체크
        if read_thread.is_alive():
            print(f"오류: 설정 파일 읽기 타임아웃 ({timeout}초). 기본값을 사용합니다.")
            return {}
        
        # 예외 발생 체크
        if exception_occurred.is_set():
            print(f"오류: 설정 파일 읽기 중 예외 발생. 기본값을 사용합니다.")
            return {}
        
        # 성공적으로 읽은 경우
        if result:
            print(f"설정 파일 로드 성공: {file_path}")
            print(f"로드된 설정값: {result}")
        else:
            print(f"경고: 설정 파일에서 유효한 설정을 찾을 수 없습니다: {file_path}")
        
        return result
        
    except Exception as e:
        print(f"오류: 설정 파일 로드 중 예외 발생: {str(e)}")
        return {}


def test_safe_loading():
    """설정 파일 안전 로딩 테스트"""
    print("=== 설정 파일 안전 로딩 테스트 ===")
    
    # 기존 설정 파일 테스트
    settings = load_settings_from_file_safe("default_settings.txt", timeout=3)
    
    if settings:
        print(f"✅ 테스트 성공: {len(settings)}개 설정값 로드됨")
        for key, value in settings.items():
            print(f"  {key}: {value} ({type(value).__name__})")
    else:
        print("❌ 테스트 실패: 설정값을 로드할 수 없음")
    
    print("================================")


if __name__ == "__main__":
    test_safe_loading()
