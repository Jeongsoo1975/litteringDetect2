# test_integrated_cache.py
"""
통합 캐시 시스템 테스트 - processing.py와의 통합 테스트
"""

import sys
import os
import time
import random
import math

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from high_performance_cache import HighPerformanceCacheManager

class MockConfig:
    """테스트용 Mock Config 클래스"""
    def __init__(self):
        self.min_size = 30
        self.max_size = 300
        self.distance_trash = 200
        self.performance_monitoring = True

def test_integrated_performance():
    """통합 성능 테스트"""
    print("=== 통합 캐시 시스템 성능 테스트 ===\n")
    
    config = MockConfig()
    cache_manager = HighPerformanceCacheManager(config)
    
    # 실제 비디오 처리 시나리오 시뮬레이션
    num_frames = 200
    num_vehicles_per_frame = 3
    num_objects_per_frame = 5
    
    print(f"시뮬레이션 조건:")
    print(f"  프레임 수: {num_frames}")
    print(f"  프레임당 차량 수: {num_vehicles_per_frame}")
    print(f"  프레임당 객체 수: {num_objects_per_frame}")
    print(f"  총 거리 계산 예상: {num_frames * num_vehicles_per_frame * num_objects_per_frame:,}회")
    
    # 차량 데이터 생성 (시간에 따라 약간씩 움직임)
    base_vehicles = []
    for i in range(num_vehicles_per_frame):
        base_x = random.randint(100, 700)
        base_y = random.randint(150, 350)
        base_vehicles.append({'base_x': base_x, 'base_y': base_y})
    
    print("\n성능 테스트 시작...")
    start_time = time.time()
    
    total_distance_checks = 0
    
    # 프레임별 처리 시뮬레이션
    for frame_idx in range(num_frames):
        # 차량 위치 업데이트 (약간의 움직임)
        vehicles = []
        for i, base_veh in enumerate(base_vehicles):
            # 차량이 시간에 따라 조금씩 움직임
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-2, 2)
            
            x1 = base_veh['base_x'] + offset_x
            y1 = base_veh['base_y'] + offset_y
            x2 = x1 + 100
            y2 = y1 + 50
            
            vehicle_box = (x1, y1, x2, y2)
            vehicles.append(vehicle_box)
        
        # 캐시에 차량 정보 업데이트
        cache_manager.update_vehicles(frame_idx, vehicles, (0, 0))
        
        # 객체들과 차량 간 거리 확인
        for obj_idx in range(num_objects_per_frame):
            obj_x = random.randint(50, 850)
            obj_y = random.randint(100, 400)
            obj_center = (obj_x, obj_y)
            
            # 거리 기반 차량 근접성 확인
            is_near_vehicle = cache_manager.check_vehicle_distance(
                obj_center, config.distance_trash, frame_idx
            )
            
            total_distance_checks += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n성능 테스트 완료!")
    print(f"  총 처리 시간: {total_time:.3f}초")
    print(f"  초당 거리 확인: {total_distance_checks/total_time:,.0f}회")
    print(f"  평균 프레임 처리 시간: {(total_time/num_frames)*1000:.2f}ms")
    
    # 캐시 성능 통계
    stats = cache_manager.get_performance_stats()
    print(f"\n캐시 성능 통계:")
    print(f"  캐시 적중률: {stats['overall_hit_rate']:.1f}%")
    print(f"  총 요청 수: {stats['total_requests']:,}")
    print(f"  거리 계산 수: {stats['distance_calculations']:,}")
    print(f"  차량 쿼리 수: {stats['vehicle_queries']:,}")
    
    distance_cache_stats = stats.get('distance_cache', {})
    print(f"  거리 캐시 엔트리: {distance_cache_stats.get('entries', 0):,}")
    print(f"  메모리 사용률: {distance_cache_stats.get('memory_usage', 0):.1f}%")
    
    # 성능 평가
    target_fps = 15  # 목표 FPS
    target_frame_time = 1.0 / target_fps  # 목표 프레임 처리 시간 (초)
    actual_frame_time = total_time / num_frames
    
    print(f"\n성능 평가:")
    print(f"  목표 FPS: {target_fps}")
    print(f"  목표 프레임 시간: {target_frame_time*1000:.1f}ms")
    print(f"  실제 프레임 시간: {actual_frame_time*1000:.1f}ms")
    
    if actual_frame_time <= target_frame_time:
        print(f"  [SUCCESS] 목표 성능 달성!")
    else:
        print(f"  [WARNING] 목표 성능 미달성 (목표의 {(target_frame_time/actual_frame_time)*100:.1f}%)")
    
    return {
        'total_time': total_time,
        'frame_time': actual_frame_time,
        'distance_checks': total_distance_checks,
        'cache_stats': stats,
        'performance_target_met': actual_frame_time <= target_frame_time
    }

def test_cache_accuracy():
    """캐시 정확성 테스트"""
    print("\n=== 캐시 정확성 테스트 ===")
    
    config = MockConfig()
    cache_manager = HighPerformanceCacheManager(config)
    
    # 테스트 케이스들
    test_cases = [
        ((100, 100), (200, 200)),  # 대각선 거리
        ((150, 150), (150, 150)),  # 같은 점
        ((0, 0), (300, 400)),      # 큰 거리
        ((250, 180), (260, 185)),  # 가까운 거리
    ]
    
    print("정확성 검증:")
    all_accurate = True
    
    for i, (p1, p2) in enumerate(test_cases):
        # 직접 계산
        direct_distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        
        # 캐시 계산
        cached_distance = cache_manager.get_optimized_distance(p1, p2)
        
        # 오차 확인
        error = abs(direct_distance - cached_distance)
        accurate = error < 0.001
        
        print(f"  테스트 {i+1}: {p1} → {p2}")
        print(f"    직접 계산: {direct_distance:.6f}")
        print(f"    캐시 계산: {cached_distance:.6f}")
        print(f"    오차: {error:.6f} {'[OK]' if accurate else '[ERROR]'}")
        
        if not accurate:
            all_accurate = False
    
    if all_accurate:
        print("\n[SUCCESS] 모든 정확성 테스트 통과!")
    else:
        print("\n[ERROR] 정확성 테스트 실패!")
    
    return all_accurate

def test_memory_efficiency():
    """메모리 효율성 테스트"""
    print("\n=== 메모리 효율성 테스트 ===")
    
    config = MockConfig()
    cache_manager = HighPerformanceCacheManager(config)
    
    # 대량의 고유한 거리 계산으로 메모리 사용량 테스트
    num_calculations = 5000
    
    print(f"대량 계산 테스트 ({num_calculations:,}회)...")
    
    for i in range(num_calculations):
        p1 = (random.randint(0, 1000), random.randint(0, 1000))
        p2 = (random.randint(0, 1000), random.randint(0, 1000))
        
        distance = cache_manager.get_optimized_distance(p1, p2)
    
    # 메모리 사용량 확인
    stats = cache_manager.get_performance_stats()
    distance_stats = stats.get('distance_cache', {})
    
    print(f"메모리 사용 결과:")
    print(f"  캐시 엔트리 수: {distance_stats.get('entries', 0):,}")
    print(f"  최대 엔트리 수: {distance_stats.get('max_entries', 0):,}")
    print(f"  메모리 사용률: {distance_stats.get('memory_usage', 0):.1f}%")
    print(f"  평균 접근 횟수: {distance_stats.get('avg_access_per_entry', 0):.1f}")
    
    # 메모리 효율성 평가
    memory_usage = distance_stats.get('memory_usage', 0)
    if memory_usage < 80:
        print(f"  [SUCCESS] 메모리 사용 효율적 (<80%)")
    else:
        print(f"  [WARNING] 메모리 사용량 높음 (>80%)")
    
    return memory_usage < 80

if __name__ == "__main__":
    try:
        print("고성능 캐시 시스템 통합 테스트 시작\n")
        
        # 1. 성능 테스트
        perf_results = test_integrated_performance()
        
        # 2. 정확성 테스트
        accuracy_passed = test_cache_accuracy()
        
        # 3. 메모리 효율성 테스트
        memory_efficient = test_memory_efficiency()
        
        # 종합 평가
        print("\n" + "="*50)
        print("종합 테스트 결과")
        print("="*50)
        
        print(f"성능 테스트: {'통과' if perf_results['performance_target_met'] else '실패'}")
        print(f"정확성 테스트: {'통과' if accuracy_passed else '실패'}")
        print(f"메모리 효율성: {'통과' if memory_efficient else '실패'}")
        
        overall_success = (perf_results['performance_target_met'] and 
                          accuracy_passed and 
                          memory_efficient)
        
        if overall_success:
            print("\n[SUCCESS] 고성능 캐시 시스템 통합 테스트 전체 통과!")
            print("실제 처리 시스템에 적용 준비 완료.")
        else:
            print("\n[WARNING] 일부 테스트 실패 - 추가 최적화 필요")
        
    except Exception as e:
        print(f"[ERROR] 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
