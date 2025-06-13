# cache_system.py
import time
import math
import threading
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Any

class FrameCache:
    """프레임 단위 데이터 캐싱 시스템"""
    
    def __init__(self, max_size: int = 100, expiry_time: float = 5.0):
        """
        Args:
            max_size: 최대 캐시 크기 (프레임 수)
            expiry_time: 캐시 만료 시간 (초)
        """
        self.max_size = max_size
        self.expiry_time = expiry_time
        self.cache = OrderedDict()  # {frame_index: cache_data}
        self.lock = threading.Lock()
        
    def _cleanup_expired(self, current_time: float):
        """만료된 캐시 엔트리 정리"""
        expired_keys = []
        for frame_index, data in self.cache.items():
            if current_time - data['timestamp'] > self.expiry_time:
                expired_keys.append(frame_index)
            else:
                break  # OrderedDict이므로 이후는 더 새로운 데이터
                
        for key in expired_keys:
            del self.cache[key]
    
    def _cleanup_size(self):
        """크기 제한에 따른 캐시 정리"""
        while len(self.cache) > self.max_size:
            # 가장 오래된 항목 제거
            self.cache.popitem(last=False)
    
    def put(self, frame_index: int, data: Any):
        """캐시에 데이터 저장"""
        with self.lock:
            current_time = time.time()
            
            # 만료된 항목 정리
            self._cleanup_expired(current_time)
            
            # 새 데이터 추가
            self.cache[frame_index] = {
                'data': data,
                'timestamp': current_time
            }
            
            # 크기 제한 적용
            self._cleanup_size()
    
    def get(self, frame_index: int) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        with self.lock:
            current_time = time.time()
            
            # 만료된 항목 정리
            self._cleanup_expired(current_time)
            
            if frame_index in self.cache:
                cache_entry = self.cache[frame_index]
                # 최근 사용된 항목을 뒤로 이동 (LRU)
                self.cache.move_to_end(frame_index)
                return cache_entry['data']
            
            return None
    
    def clear(self):
        """캐시 비우기"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        with self.lock:
            current_time = time.time()
            valid_entries = 0
            for data in self.cache.values():
                if current_time - data['timestamp'] <= self.expiry_time:
                    valid_entries += 1
            
            return {
                'total_entries': len(self.cache),
                'valid_entries': valid_entries,
                'max_size': self.max_size,
                'memory_usage_percent': (len(self.cache) / self.max_size) * 100
            }


class VehicleCache:
    """차량 정보 전용 캐싱 시스템"""
    
    def __init__(self, max_frames: int = 50):
        """
        Args:
            max_frames: 캐시할 최대 프레임 수
        """
        self.max_frames = max_frames
        self.vehicle_history = OrderedDict()  # {frame_index: vehicle_data}
        self.vehicle_tracking = {}  # {vehicle_id: last_position}
        self.lock = threading.Lock()
        
    def update_vehicles(self, frame_index: int, yolo_boxes: List[Tuple], roi_offset: Tuple[int, int]):
        """차량 정보 업데이트 및 캐싱"""
        with self.lock:
            vehicle_data = []
            roi_x, roi_y = roi_offset
            
            for box in yolo_boxes:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                vehicle_info = {
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'area': (x2 - x1) * (y2 - y1),
                    'roi_bbox': (x1 - roi_x, y1 - roi_y, x2 - roi_x, y2 - roi_y)
                }
                vehicle_data.append(vehicle_info)
            
            # 프레임별 차량 데이터 저장
            self.vehicle_history[frame_index] = vehicle_data
            
            # 크기 제한 적용
            while len(self.vehicle_history) > self.max_frames:
                self.vehicle_history.popitem(last=False)
    
    def get_vehicles(self, frame_index: int) -> Optional[List[Dict]]:
        """특정 프레임의 차량 정보 조회"""
        with self.lock:
            return self.vehicle_history.get(frame_index)
    
    def get_recent_vehicles(self, current_frame: int, lookback_frames: int = 5) -> List[Dict]:
        """최근 프레임들의 차량 정보 통합 조회"""
        with self.lock:
            recent_vehicles = []
            
            for i in range(max(0, current_frame - lookback_frames), current_frame + 1):
                vehicles = self.vehicle_history.get(i, [])
                recent_vehicles.extend(vehicles)
            
            return recent_vehicles
    
    def clear(self):
        """캐시 비우기"""
        with self.lock:
            self.vehicle_history.clear()
            self.vehicle_tracking.clear()


class DistanceCache:
    """거리 계산 결과 캐싱 시스템"""
    
    def __init__(self, max_entries: int = 1000, expiry_time: float = 3.0):
        """
        Args:
            max_entries: 최대 캐시 엔트리 수
            expiry_time: 캐시 만료 시간 (초)
        """
        self.max_entries = max_entries
        self.expiry_time = expiry_time
        self.distance_cache = OrderedDict()  # {cache_key: result}
        self.lock = threading.Lock()
    
    def _generate_key(self, obj_center: Tuple[int, int], vehicle_center: Tuple[int, int]) -> str:
        """거리 계산을 위한 캐시 키 생성 (최적화됨)"""
        # 더 큰 정밀도 단위로 반올림하여 캐시 적중률 향상
        precision = 10  # 10픽셀 단위로 반올림 (캐시 적중률 향상)
        
        obj_x = (obj_center[0] // precision) * precision
        obj_y = (obj_center[1] // precision) * precision
        veh_x = (vehicle_center[0] // precision) * precision
        veh_y = (vehicle_center[1] // precision) * precision
        
        # 더 빠른 키 생성 (문자열 대신 튜플 사용)
        return (obj_x, obj_y, veh_x, veh_y)
    
    def get_distance(self, obj_center: Tuple[int, int], vehicle_center: Tuple[int, int]) -> float:
        """캐싱된 거리 계산 또는 새로 계산하여 캐싱 (성능 최적화)"""
        # 빠른 사전 검사 - 같은 위치면 0 반환
        if obj_center == vehicle_center:
            return 0.0
            
        cache_key = self._generate_key(obj_center, vehicle_center)
        
        # 캐시된 결과 확인 (락 없이 빠른 체크)
        if cache_key in self.distance_cache:
            cached_entry = self.distance_cache[cache_key]
            current_time = time.time()
            if current_time - cached_entry['timestamp'] <= self.expiry_time:
                # 캐시 적중 - 최근 사용 항목 업데이트는 생략 (성능 우선)
                return cached_entry['distance']
        
        # 새로 계산 (가장 비용이 큰 부분)
        distance = math.sqrt(
            (obj_center[0] - vehicle_center[0]) ** 2 + 
            (obj_center[1] - vehicle_center[1]) ** 2
        )
        
        # 캐시에 저장 (비동기적으로 수행하거나 배치로 처리 가능)
        with self.lock:
            current_time = time.time()
            self.distance_cache[cache_key] = {
                'distance': distance,
                'timestamp': current_time
            }
            
            # 크기 제한 적용 (간단한 방식)
            if len(self.distance_cache) > self.max_entries:
                # 오래된 항목들을 일괄 제거 (더 효율적)
                items_to_remove = len(self.distance_cache) - self.max_entries + 10
                keys_to_remove = list(self.distance_cache.keys())[:items_to_remove]
                for key in keys_to_remove:
                    del self.distance_cache[key]
        
        return distance
    
    def clear(self):
        """캐시 비우기"""
        with self.lock:
            self.distance_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        with self.lock:
            current_time = time.time()
            valid_entries = 0
            
            for entry in self.distance_cache.values():
                if current_time - entry['timestamp'] <= self.expiry_time:
                    valid_entries += 1
            
            return {
                'total_entries': len(self.distance_cache),
                'valid_entries': valid_entries,
                'max_entries': self.max_entries,
                'hit_rate_estimate': valid_entries / max(1, len(self.distance_cache))
            }


class OptimizedCacheManager:
    """통합 캐시 관리자"""
    
    def __init__(self, config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        
        # 개별 캐시 시스템들
        self.frame_cache = FrameCache(max_size=100, expiry_time=5.0)
        self.vehicle_cache = VehicleCache(max_frames=50)
        self.distance_cache = DistanceCache(max_entries=1000, expiry_time=3.0)
        
        # 통계 추적
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
        
    def get_optimized_vehicle_distance(self, obj_center: Tuple[int, int], 
                                     frame_index: int, 
                                     current_vehicles: Optional[List[Dict]] = None) -> Tuple[float, Optional[Dict]]:
        """최적화된 차량 거리 계산"""
        min_distance = float('inf')
        closest_vehicle = None
        
        # 현재 프레임 차량 정보가 없으면 캐시에서 조회
        if current_vehicles is None:
            current_vehicles = self.vehicle_cache.get_vehicles(frame_index)
            if current_vehicles is None:
                # 최근 프레임들에서 차량 정보 조회
                current_vehicles = self.vehicle_cache.get_recent_vehicles(frame_index)
                self.cache_misses['vehicle'] += 1
            else:
                self.cache_hits['vehicle'] += 1
        
        # 각 차량과의 거리 계산 (캐싱 적용)
        for vehicle in current_vehicles or []:
            vehicle_center = vehicle['center']
            
            # 캐싱된 거리 계산 사용
            distance = self.distance_cache.get_distance(obj_center, vehicle_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_vehicle = vehicle
        
        return min_distance, closest_vehicle
    
    def update_caches(self, frame_index: int, yolo_boxes: List[Tuple], roi_offset: Tuple[int, int]):
        """모든 캐시 업데이트"""
        # 차량 캐시 업데이트
        self.vehicle_cache.update_vehicles(frame_index, yolo_boxes, roi_offset)
    
    def clear_all_caches(self):
        """모든 캐시 비우기"""
        self.frame_cache.clear()
        self.vehicle_cache.clear()
        self.distance_cache.clear()
        self.cache_hits.clear()
        self.cache_misses.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """캐시 성능 통계 반환"""
        frame_stats = self.frame_cache.get_stats()
        distance_stats = self.distance_cache.get_stats()
        
        total_hits = sum(self.cache_hits.values())
        total_misses = sum(self.cache_misses.values())
        total_requests = total_hits + total_misses
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'overall_hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_hits': dict(self.cache_hits),
            'cache_misses': dict(self.cache_misses),
            'frame_cache': frame_stats,
            'distance_cache': distance_stats,
            'memory_efficiency': {
                'frame_cache_usage': frame_stats['memory_usage_percent'],
                'distance_cache_entries': distance_stats['total_entries']
            }
        }
