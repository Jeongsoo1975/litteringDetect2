# test_settings_loading.py - 설정 파일 로딩 타임아웃 문제 테스트

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing import load_settings_from_file, Config
import time

def test_settings_loading():
    """설정 파일 로딩 테스트"""
    print("=== 설정 파일 로딩 테스트 시작 ===")
    
    # 1. 기본 설정 파일 로딩 테스트
    print("\n1. 기본 설정 파일 로딩 테스트")
    start_time = time.time()
    
    try:
        settings = load_settings_from_file("default_settings.txt", timeout=3)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 소요 시간: {elapsed:.2f}초")
        
        if settings:
            print(f"✅ 성공: {len(settings)}개 설정값 로드됨")
            print("주요 설정값:")
            key_settings = ['min_size', 'max_size', 'yolo_confidence_value', 'batch_size', 'debug_detection']
            for key in key_settings:
                if key in settings:
                    print(f"  {key}: {settings[key]} ({type(settings[key]).__name__})")
        else:
            print("❌ 실패: 설정값을 로드할 수 없음")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 예외 발생 ({elapsed:.2f}초): {str(e)}")
    
    # 2. Config 클래스 초기화 테스트
    print("\n2. Config 클래스 초기화 테스트")
    start_time = time.time()
    
    try:
        config = Config()
        elapsed = time.time() - start_time
        
        print(f"⏱️ 소요 시간: {elapsed:.2f}초")
        print(f"✅ Config 초기화 성공")
        print("Config 주요 값:")
        print(f"  min_size: {config.min_size}")
        print(f"  max_size: {config.max_size}")
        print(f"  yolo_confidence_value: {config.yolo_confidence_value}")
        print(f"  debug_detection: {config.debug_detection}")
        print(f"  performance_monitoring: {config.performance_monitoring}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Config 초기화 실패 ({elapsed:.2f}초): {str(e)}")
    
    # 3. 존재하지 않는 파일 테스트
    print("\n3. 존재하지 않는 파일 테스트")
    start_time = time.time()
    
    try:
        settings = load_settings_from_file("nonexistent_file.txt", timeout=2)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 소요 시간: {elapsed:.2f}초")
        
        if not settings:
            print("✅ 예상대로 빈 딕셔너리 반환됨")
        else:
            print("❌ 예상과 다름: 설정값이 반환됨")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 예외 발생 ({elapsed:.2f}초): {str(e)}")
    
    print("\n=== 설정 파일 로딩 테스트 완료 ===")

if __name__ == "__main__":
    test_settings_loading()
