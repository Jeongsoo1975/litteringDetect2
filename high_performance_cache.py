# high_performance_cache.py
"""
고성능 캐시 시스템 - 실제 비디오 처리에 최적화
"""

import time
import math
import threading
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

class FastDistanceCache:
    """고성능 거리 캐시 (메모리 효율적)"""
    
    def __init__(self, precision: int = 15):
        """
        Args:
            precision: 캐시 키 정밀도 (픽셀 단위)
        """
        self.precision = precision
        self.cache = {}  # 단순 딕셔너리 사용 (OrderedDict 오버헤드 제거)
        self.access_count = defaultdict(int)  # 접근 횟수 추적
        self.max_entries = 1000  # 메모리 효율성을 위해 크기 감소
        
    def _make_key(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """빠른 캐시 키 생성"""
        # 정밀도에 따른 반올림으로 유사한 좌표들을 같은 키로 만듦
        x1 = (p1[0] // self.precision) * self.precision
        y1 = (p1[1] // self.precision) * self.precision
        x2 = (p2[0] // self.precision) * self.precision
        y2 = (p2[1] // self.precision) * self.precision
        
        # 좌표 순서를 정규화하여 (A,B)와 (B,A)가 같은 키를 갖도록 함
        if (x1, y1) > (x2, y2):
            return (x2, y2, x1, y1)
        else:
            return (x1, y1, x2, y2)
    
    def get_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """캐시를 활용한 빠른 거리 계산"""
        # 같은 점이면 0 반환
        if p1 == p2:
            return 0.0
        
        key = self._make_key(p1, p2)
        
        # 캐시 확인
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        
        # 계산 및 캐시 저장
        distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        
        # 캐시 크기 관리
        if len(self.cache) >= self.max_entries:
            self._cleanup_cache()
        
        self.cache[key] = distance
        self.access_count[key] = 1
        
        return distance
    
    def _cleanup_cache(self):
        """LFU (Least Frequently Used) 방식으로 캐시 정리 - 적극적 정리"""
        # 접근 횟수가 가장 적은 항목들 제거
        sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])
        items_to_remove = len(self.cache) // 3  # 33% 제거 (더 적극적)
        
        for key, _ in sorted_items[:items_to_remove]:
            if key in self.cache:
                del self.cache[key]
            del self.access_count[key]
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()
        self.access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_access = sum(self.access_count.values())
        return {
            'entries': len(self.cache),
            'max_entries': self.max_entries,
            'total_access': total_access,
            'avg_access_per_entry': total_access / max(1, len(self.cache)),
            'memory_usage': len(self.cache) / self.max_entries * 100
        }


class SmartVehicleCache:
    """지능형 차량 정보 캐시"""
    
    def __init__(self, max_frames: int = 30):
        """
        Args:
            max_frames: 캐시할 최대 프레임 수
        """
        self.max_frames = max_frames
        self.vehicle_data = {}  # {frame_index: vehicle_list}
        self.current_frame = 0
        
    def update(self, frame_index: int, vehicles: List[Dict]) -> None:
        """차량 정보 업데이트"""
        self.current_frame = frame_index
        self.vehicle_data[frame_index] = vehicles
        
        # 오래된 프레임 데이터 정리
        if len(self.vehicle_data) > self.max_frames:
            old_frames = sorted(self.vehicle_data.keys())[:-self.max_frames]
            for old_frame in old_frames:
                del self.vehicle_data[old_frame]
    
    def get_recent_vehicles(self, lookback: int = 3) -> List[Dict]:
        """최근 프레임들의 차량 정보 통합"""
        vehicles = []
        start_frame = max(0, self.current_frame - lookback)
        
        for frame_idx in range(start_frame, self.current_frame + 1):
            if frame_idx in self.vehicle_data:
                vehicles.extend(self.vehicle_data[frame_idx])
        
        return vehicles
    
    def get_current_vehicles(self) -> List[Dict]:
        """현재 프레임 차량 정보"""
        return self.vehicle_data.get(self.current_frame, [])
    
    def clear(self):
        """캐시 초기화"""
        self.vehicle_data.clear()
        self.current_frame = 0


class HighPerformanceCacheManager:
    """고성능 통합 캐시 관리자"""
    
    def __init__(self, config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        
        # 고성능 캐시 시스템들
        self.distance_cache = FastDistanceCache(precision=20)  # 20픽셀 정밀도
        self.vehicle_cache = SmartVehicleCache(max_frames=50)
        
        # 성능 통계
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'distance_calculations': 0,
            'vehicle_queries': 0
        }
        
        # 성능 모니터링
        self.last_stats_time = time.time()
        self.stats_interval = 10.0  # 10초마다 통계 리셋
        
    def get_optimized_distance(self, obj_center: Tuple[int, int], 
                             vehicle_center: Tuple[int, int]) -> float:
        """최적화된 거리 계산"""
        self.stats['distance_calculations'] += 1
        
        # 캐시된 거리 계산 사용
        distance = self.distance_cache.get_distance(obj_center, vehicle_center)
        
        return distance
    
    def find_closest_vehicle(self, obj_center: Tuple[int, int], 
                           frame_index: Optional[int] = None) -> Tuple[float, Optional[Dict]]:
        """가장 가까운 차량 찾기"""
        self.stats['vehicle_queries'] += 1
        
        # 차량 정보 가져오기
        if frame_index is not None:
            vehicles = self.vehicle_cache.get_recent_vehicles(lookback=2)
        else:
            vehicles = self.vehicle_cache.get_current_vehicles()
        
        if not vehicles:
            self.stats['cache_misses'] += 1
            return float('inf'), None
        
        self.stats['cache_hits'] += 1
        
        min_distance = float('inf')
        closest_vehicle = None
        
        # 각 차량과의 거리 계산 (캐시 활용)
        for vehicle in vehicles:
            vehicle_center = vehicle.get('center', (0, 0))
            distance = self.get_optimized_distance(obj_center, vehicle_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_vehicle = vehicle
        
        return min_distance, closest_vehicle
    
    def update_vehicles(self, frame_index: int, yolo_boxes: List[Tuple], 
                       roi_offset: Tuple[int, int]):
        """차량 정보 업데이트"""
        roi_x, roi_y = roi_offset
        vehicles = []
        
        for box in yolo_boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            vehicle_info = {
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'area': (x2 - x1) * (y2 - y1)
            }
            vehicles.append(vehicle_info)
        
        self.vehicle_cache.update(frame_index, vehicles)
    
    def check_vehicle_distance(self, obj_center: Tuple[int, int], 
                             max_distance: float,
                             frame_index: Optional[int] = None) -> bool:
        """차량과의 거리가 임계값 이내인지 확인"""
        min_distance, closest_vehicle = self.find_closest_vehicle(obj_center, frame_index)
        return min_distance <= max_distance
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        current_time = time.time()
        
        # 주기적으로 통계 리셋
        if current_time - self.last_stats_time > self.stats_interval:
            self._reset_periodic_stats()
            self.last_stats_time = current_time
        
        # 캐시별 통계 수집
        distance_stats = self.distance_cache.get_stats()
        
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / max(1, total_requests)) * 100
        
        return {
            'overall_hit_rate': hit_rate,
            'total_requests': total_requests,
            'distance_calculations': self.stats['distance_calculations'],
            'vehicle_queries': self.stats['vehicle_queries'],
            'distance_cache': distance_stats,
            'vehicle_cache_frames': len(self.vehicle_cache.vehicle_data),
            'performance_summary': {
                'cache_efficiency': hit_rate,
                'memory_usage': distance_stats['memory_usage'],
                'avg_access_per_entry': distance_stats['avg_access_per_entry']
            }
        }
    
    def _reset_periodic_stats(self):
        """주기적 통계 리셋"""
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'distance_calculations': 0,
            'vehicle_queries': 0
        }
    
    def clear_all_caches(self):
        """모든 캐시 초기화"""
        self.distance_cache.clear()
        self.vehicle_cache.clear()
        self._reset_periodic_stats()


# 편의 함수들
def create_high_performance_cache(config) -> HighPerformanceCacheManager:
    """고성능 캐시 관리자 팩토리 함수"""
    return HighPerformanceCacheManager(config)


def test_high_performance_cache():
    """고성능 캐시 시스템 테스트"""
    print("=== 고성능 캐시 시스템 테스트 ===\n")
    
    class MockConfig:
        distance_trash = 200
    
    config = MockConfig()
    cache_manager = HighPerformanceCacheManager(config)
    
    # 테스트 데이터
    vehicles = [
        {'bbox': (100, 100, 200, 150), 'center': (150, 125)},
        {'bbox': (300, 200, 400, 250), 'center': (350, 225)},
        {'bbox': (500, 150, 600, 200), 'center': (550, 175)}
    ]
    
    # 차량 정보 업데이트
    for frame in range(10):
        vehicle_boxes = [(v['bbox'][0], v['bbox'][1], v['bbox'][2], v['bbox'][3]) for v in vehicles]
        cache_manager.update_vehicles(frame, vehicle_boxes, (0, 0))
    
    # 거리 계산 테스트
    test_objects = [(160, 130), (360, 230), (560, 180), (700, 300)]
    
    print("거리 계산 테스트:")
    for i, obj_pos in enumerate(test_objects):
        min_dist, closest_veh = cache_manager.find_closest_vehicle(obj_pos)
        within_threshold = cache_manager.check_vehicle_distance(obj_pos, config.distance_trash)
        
        print(f"  객체 {i+1} ({obj_pos[0]}, {obj_pos[1]}): "
              f"최근접 거리={min_dist:.1f}, 임계값 내={'예' if within_threshold else '아니오'}")
    
    # 성능 통계
    stats = cache_manager.get_performance_stats()
    print(f"\n성능 통계:")
    print(f"  캐시 적중률: {stats['overall_hit_rate']:.1f}%")
    print(f"  총 요청 수: {stats['total_requests']}")
    print(f"  거리 계산 수: {stats['distance_calculations']}")
    print(f"  메모리 사용률: {stats['distance_cache']['memory_usage']:.1f}%")
    
    return cache_manager


if __name__ == "__main__":
    cache_manager = test_high_performance_cache()
