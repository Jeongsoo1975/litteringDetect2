# test_cache_system.py
"""
캐시 시스템 성능 테스트 스크립트
"""

import time
import random
import math
from cache_system import OptimizedCacheManager, FrameCache, VehicleCache, DistanceCache

class MockConfig:
    """테스트용 Mock Config 클래스"""
    def __init__(self):
        self.min_size = 30
        self.max_size = 300
        self.distance_trash = 200
        self.performance_monitoring = True

def generate_test_data():
    """테스트용 데이터 생성"""
    # 가상의 차량 정보 생성
    vehicles = []
    for i in range(5):  # 5대의 차량
        x1 = random.randint(100, 500)
        y1 = random.randint(100, 300)
        x2 = x1 + random.randint(80, 150)
        y2 = y1 + random.randint(40, 80)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        vehicle = {
            'bbox': (x1, y1, x2, y2),
            'center': (center_x, center_y),
            'area': (x2 - x1) * (y2 - y1)
        }
        vehicles.append(vehicle)
    
    return vehicles

def test_cache_performance():
    """캐시 성능 테스트"""
    print("=== 캐시 시스템 성능 테스트 시작 ===\n")
    
    config = MockConfig()
    cache_manager = OptimizedCacheManager(config)
    
    # 테스트 데이터 생성
    vehicles = generate_test_data()
    print(f"생성된 테스트 차량 수: {len(vehicles)}")
    
    # 테스트 객체 위치들
    test_objects = []
    for i in range(20):  # 20개의 테스트 객체
        x = random.randint(50, 600)
        y = random.randint(50, 400)
        test_objects.append((x, y))
    
    print(f"생성된 테스트 객체 수: {len(test_objects)}")
    print()
    
    # 1. 캐시 없이 거리 계산 (기존 방식)
    print("1. 캐시 없이 거리 계산 테스트...")
    start_time = time.time()
    
    for frame_idx in range(100):  # 100 프레임 시뮬레이션
        for obj_center in test_objects:
            for vehicle in vehicles:
                veh_center = vehicle['center']
                # 기존 방식의 거리 계산
                distance = math.sqrt(
                    (obj_center[0] - veh_center[0]) ** 2 + 
                    (obj_center[1] - veh_center[1]) ** 2
                )
    
    no_cache_time = time.time() - start_time
    total_calculations = 100 * len(test_objects) * len(vehicles)
    print(f"   소요 시간: {no_cache_time:.4f}초")
    print(f"   총 계산 수: {total_calculations:,}회")
    print(f"   초당 계산 수: {total_calculations/no_cache_time:,.0f}회")
    print()
    
    # 2. 캐시를 사용한 거리 계산
    print("2. 캐시를 사용한 거리 계산 테스트...")
    start_time = time.time()
    
    for frame_idx in range(100):  # 100 프레임 시뮬레이션
        # 차량 정보를 캐시에 업데이트
        cache_manager.update_caches(
            frame_index=frame_idx,
            yolo_boxes=[(v['bbox'][0], v['bbox'][1], v['bbox'][2], v['bbox'][3]) for v in vehicles],
            roi_offset=(0, 0)
        )
        
        for obj_center in test_objects:
            # 캐시를 활용한 최적화된 거리 계산
            min_distance, closest_vehicle = cache_manager.get_optimized_vehicle_distance(
                obj_center=obj_center,
                frame_index=frame_idx,
                current_vehicles=vehicles
            )
    
    cache_time = time.time() - start_time
    print(f"   소요 시간: {cache_time:.4f}초")
    print(f"   총 계산 수: {total_calculations:,}회")
    print(f"   초당 계산 수: {total_calculations/cache_time:,.0f}회")
    print()
    
    # 3. 성능 개선 비교
    print("3. 성능 비교 결과")
    improvement = (no_cache_time - cache_time) / no_cache_time * 100
    speedup = no_cache_time / cache_time
    
    print(f"   캐시 없이: {no_cache_time:.4f}초")
    print(f"   캐시 사용: {cache_time:.4f}초")
    print(f"   성능 개선: {improvement:.1f}%")
    print(f"   속도 향상: {speedup:.2f}배")
    print()
    
    # 4. 캐시 통계 확인
    print("4. 캐시 성능 통계")
    stats = cache_manager.get_performance_stats()
    
    print(f"   전체 캐시 적중률: {stats['overall_hit_rate']:.1f}%")
    print(f"   총 요청 수: {stats['total_requests']:,}")
    print(f"   캐시 적중: {stats['cache_hits']}")
    print(f"   캐시 미스: {stats['cache_misses']}")
    
    if 'distance_cache' in stats:
        distance_stats = stats['distance_cache']
        print(f"   거리 캐시 엔트리: {distance_stats['total_entries']:,}")
        print(f"   거리 캐시 적중률: {distance_stats['hit_rate_estimate']:.1%}")
    
    print()
    
    # 5. 메모리 효율성 테스트
    print("5. 메모리 효율성 테스트")
    if 'memory_efficiency' in stats:
        memory_stats = stats['memory_efficiency']
        print(f"   프레임 캐시 사용률: {memory_stats.get('frame_cache_usage', 0):.1f}%")
        print(f"   거리 캐시 엔트리 수: {memory_stats.get('distance_cache_entries', 0):,}")
    
    print()
    
    # 6. 캐시 정확성 검증
    print("6. 캐시 정확성 검증")
    verification_passed = True
    test_pairs = [(test_objects[0], vehicles[0]['center']) for _ in range(5)]
    
    for obj_pos, veh_pos in test_pairs:
        # 직접 계산
        direct_distance = math.sqrt(
            (obj_pos[0] - veh_pos[0]) ** 2 + (obj_pos[1] - veh_pos[1]) ** 2
        )
        
        # 캐시 계산
        cached_distance = cache_manager.distance_cache.get_distance(obj_pos, veh_pos)
        
        # 오차 검사 (부동소수점 오차 허용)
        if abs(direct_distance - cached_distance) > 0.001:
            print(f"   [X] 오차 발견: 직접계산={direct_distance:.6f}, 캐시계산={cached_distance:.6f}")
            verification_passed = False
    
    if verification_passed:
        print(f"   [O] 캐시 정확성 검증 통과")
    else:
        print(f"   [X] 캐시 정확성 검증 실패")
    
    print()
    
    return {
        'no_cache_time': no_cache_time,
        'cache_time': cache_time,
        'improvement_percent': improvement,
        'speedup_factor': speedup,
        'cache_stats': stats,
        'accuracy_verified': verification_passed
    }

def test_cache_expiry():
    """캐시 만료 기능 테스트"""
    print("=== 캐시 만료 기능 테스트 ===\n")
    
    # 짧은 만료 시간으로 테스트
    distance_cache = DistanceCache(max_entries=10, expiry_time=0.1)  # 0.1초 만료
    
    # 테스트 데이터 추가
    test_point1 = (100, 100)
    test_point2 = (200, 200)
    
    print("1. 캐시에 데이터 추가...")
    distance1 = distance_cache.get_distance(test_point1, test_point2)
    print(f"   계산된 거리: {distance1:.2f}")
    
    # 캐시 통계 확인
    stats = distance_cache.get_stats()
    print(f"   캐시 엔트리 수: {stats['total_entries']}")
    print(f"   유효 엔트리 수: {stats['valid_entries']}")
    
    print("\n2. 0.15초 대기 (만료 시간 초과)...")
    time.sleep(0.15)
    
    # 만료 후 다시 거리 계산
    print("3. 만료 후 재계산...")
    distance2 = distance_cache.get_distance(test_point1, test_point2)
    print(f"   재계산된 거리: {distance2:.2f}")
    
    # 캐시 통계 재확인
    stats_after = distance_cache.get_stats()
    print(f"   캐시 엔트리 수: {stats_after['total_entries']}")
    print(f"   유효 엔트리 수: {stats_after['valid_entries']}")
    
    # 정확성 확인
    if abs(distance1 - distance2) < 0.001:
        print("   [O] 만료 후에도 계산 결과 일치")
    else:
        print("   [X] 만료 후 계산 결과 불일치")
    
    print()

def stress_test_cache():
    """캐시 시스템 스트레스 테스트"""
    print("=== 캐시 시스템 스트레스 테스트 ===\n")
    
    config = MockConfig()
    cache_manager = OptimizedCacheManager(config)
    
    # 대량의 데이터로 스트레스 테스트
    num_vehicles = 20  # 20대의 차량
    num_objects = 100  # 100개의 객체
    num_frames = 500   # 500 프레임
    
    print(f"테스트 조건:")
    print(f"   차량 수: {num_vehicles}")
    print(f"   객체 수: {num_objects}")
    print(f"   프레임 수: {num_frames}")
    print(f"   총 계산 수: {num_vehicles * num_objects * num_frames:,}")
    
    # 테스트 데이터 생성
    vehicles = []
    for i in range(num_vehicles):
        x1 = random.randint(50, 800)
        y1 = random.randint(50, 400)
        vehicle = {
            'bbox': (x1, y1, x1 + 100, y1 + 50),
            'center': (x1 + 50, y1 + 25)
        }
        vehicles.append(vehicle)
    
    objects = [(random.randint(0, 900), random.randint(0, 500)) for _ in range(num_objects)]
    
    print("\n스트레스 테스트 진행 중...")
    start_time = time.time()
    
    for frame_idx in range(num_frames):
        # 차량 정보 업데이트
        cache_manager.update_caches(
            frame_index=frame_idx,
            yolo_boxes=[(v['bbox'][0], v['bbox'][1], v['bbox'][2], v['bbox'][3]) for v in vehicles],
            roi_offset=(0, 0)
        )
        
        # 모든 객체-차량 조합의 거리 계산
        for obj_center in objects:
            for vehicle in vehicles:
                veh_center = vehicle['center']
                distance = cache_manager.distance_cache.get_distance(obj_center, veh_center)
    
    end_time = time.time()
    total_time = end_time - start_time
    total_calculations = num_vehicles * num_objects * num_frames
    
    print(f"\n스트레스 테스트 완료!")
    print(f"   총 소요 시간: {total_time:.2f}초")
    print(f"   초당 계산 수: {total_calculations/total_time:,.0f}회")
    
    # 최종 캐시 통계
    final_stats = cache_manager.get_performance_stats()
    print(f"   최종 캐시 적중률: {final_stats['overall_hit_rate']:.1f}%")
    
    if final_stats['overall_hit_rate'] > 50:
        print("   [O] 스트레스 테스트 통과 (적중률 > 50%)")
    else:
        print("   [!] 스트레스 테스트 주의 (적중률 낮음)")
    
    print()
    return final_stats

if __name__ == "__main__":
    try:
        # 1. 기본 성능 테스트
        results = test_cache_performance()
        
        # 2. 캐시 만료 테스트
        test_cache_expiry()
        
        # 3. 스트레스 테스트
        stress_results = stress_test_cache()
        
        # 종합 결과 출력
        print("=== 종합 테스트 결과 ===")
        print(f"기본 성능 향상: {results['improvement_percent']:.1f}%")
        print(f"속도 향상 배수: {results['speedup_factor']:.2f}배")
        print(f"정확성 검증: {'통과' if results['accuracy_verified'] else '실패'}")
        print(f"스트레스 테스트 적중률: {stress_results['overall_hit_rate']:.1f}%")
        
        if (results['improvement_percent'] > 10 and 
            results['accuracy_verified'] and 
            stress_results['overall_hit_rate'] > 30):
            print("\n[SUCCESS] 캐시 시스템 테스트 전체 통과!")
        else:
            print("\n[WARNING] 캐시 시스템 개선이 필요합니다.")
            
    except Exception as e:
        print(f"[ERROR] 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
