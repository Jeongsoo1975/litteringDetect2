# detection_strategies.py

##########################
# 감지 전략 클래스들
##########################
from abc import ABC, abstractmethod
import numpy as np
import logging
import math  # 추가
import traceback  # 추가

# 로깅 설정
logger = logging.getLogger(__name__)


class DetectionStrategy(ABC):
    """쓰레기 감지 전략의 기본 클래스"""

    @abstractmethod
    def name(self):
        """전략 이름 반환"""
        pass

    @abstractmethod
    def description(self):
        """전략 설명 반환"""
        pass

    @abstractmethod
    def check(self, frame, tracking_info, config, vehicle_info=None):
        """감지 조건 확인"""
        pass


class VehicleOverlapStrategy(DetectionStrategy):
    """차량과 오브젝트 겹침 확인 전략"""

    def name(self):
        return "차량 비겹침 확인"

    def description(self):
        return "선정된 오브젝트가 차량 바운딩박스와 겹치면 선정 배제"

    def check(self, frame, tracking_info, config, vehicle_info=None):
        logger.debug(
            f"[{self.name()}] 전략 검사 시작: tracking_info={len(tracking_info) if tracking_info else 0}, vehicle_info={len(vehicle_info) if vehicle_info else 0}")

        if not vehicle_info or not tracking_info:
            logger.debug(f"[{self.name()}] 차량 정보 또는 추적 정보 없음 - 통과")
            return True  # 차량 정보가 없으면 항상 통과

        # 현재 추적 중인 오브젝트의 바운딩 박스
        if len(tracking_info) < 1:
            logger.debug(f"[{self.name()}] 추적 정보 부족 - 통과")
            return True

        obj_bbox = tracking_info[-1]['bbox']  # (x1, y1, x2, y2)
        obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
        obj_center_y = (obj_bbox[1] + obj_bbox[3]) / 2

        # 차량의 바운딩 박스
        for vehicle in vehicle_info:
            veh_bbox = vehicle['bbox']  # (x1, y1, x2, y2)

            # 객체의 중심점이 차량 바운딩 박스 내부에 있는지 확인
            if (veh_bbox[0] <= obj_center_x <= veh_bbox[2] and
                    veh_bbox[1] <= obj_center_y <= veh_bbox[3]):
                logger.debug(f"[{self.name()}] 객체 중심점이 차량 내부에 있음 - 배제")
                return False  # 차량 내부에 있으면 쓰레기가 아님

            # 두 바운딩 박스 간의 겹침 확인
            intersection_area = self._calculate_intersection_area(obj_bbox, veh_bbox)
            obj_area = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])

            # 겹치는 부분이 있으면 쓰레기가 아님
            if intersection_area > 0 and intersection_area / obj_area > config.vehicle_overlap_threshold:
                logger.debug(
                    f"[{self.name()}] 차량과 겹침 비율 높음 ({intersection_area / obj_area:.3f} > {config.vehicle_overlap_threshold}) - 배제")
                return False

        logger.debug(f"[{self.name()}] 차량과 겹치지 않음 - 통과")
        return True

    def _calculate_intersection_area(self, bbox1, bbox2):
        """두 바운딩 박스의 교차 영역 면적 계산"""
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])

        width = max(0, x2_min - x1_max)
        height = max(0, y2_min - y1_max)

        return width * height


class SizeRangeStrategy(DetectionStrategy):
    """오브젝트 크기 범위 확인 전략"""

    def name(self):
        return "크기 범위 확인"

    def description(self):
        return "선정된 오브젝트의 픽셀 수(크기)가, 설정된 최소/최대 범위 이내인지 확인"

    def check(self, frame, tracking_info, config, vehicle_info=None):
        logger.debug(f"[{self.name()}] 전략 검사 시작: tracking_info={len(tracking_info) if tracking_info else 0}")

        if not tracking_info or len(tracking_info) < 1:
            logger.debug(f"[{self.name()}] 추적 정보 없음 - 배제")
            return False

        # 현재 오브젝트의 크기 (면적) 계산
        obj_bbox = tracking_info[-1]['bbox']  # (x1, y1, x2, y2)
        width = obj_bbox[2] - obj_bbox[0]
        height = obj_bbox[3] - obj_bbox[1]
        area = width * height

        # 최소/최대 크기 범위 내에 있는지 확인
        result = config.min_size <= area <= config.max_size
        if result:
            logger.debug(f"[{self.name()}] 크기 범위 내 (면적: {area}, 범위: {config.min_size}-{config.max_size}) - 통과")
        else:
            logger.debug(f"[{self.name()}] 크기 범위 밖 (면적: {area}, 범위: {config.min_size}-{config.max_size}) - 배제")

        return result


class VehicleDistanceStrategy(DetectionStrategy):
    """차량과 오브젝트 간의 거리 확인 전략"""

    def name(self):
        return "차량 거리 확인"

    def description(self):
        return "선정된 오브젝트와 차량 사이의 간격이 일정 범위 이내인지 확인"

    def check(self, frame, tracking_info, config, vehicle_info=None):
        """차량과 쓰레기 객체 간의 연관성 확인"""
        logger.debug(
            f"[{self.name()}] 전략 검사 시작: tracking_info={len(tracking_info) if tracking_info else 0}, vehicle_info={len(vehicle_info) if vehicle_info else 0}")

        if not vehicle_info or not tracking_info:
            logger.debug(f"[{self.name()}] 차량 정보 또는 추적 정보 없음 - 배제")
            return False

        # 가장 최근 위치
        obj_bbox = tracking_info[-1]['bbox']  # (x1, y1, x2, y2)
        obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
        obj_center_y = (obj_bbox[1] + obj_bbox[3]) / 2

        # 각 차량과의 거리 계산
        min_distance = float('inf')
        closest_edge = None

        for vehicle in vehicle_info:
            veh_bbox = vehicle['bbox']  # (x1, y1, x2, y2)
            vbx1, vby1, vbx2, vby2 = veh_bbox

            # 객체 중심점과 각 경계선 간의 거리 계산
            # 객체가 차량의 왼쪽에 있는 경우
            if obj_center_x < vbx1:
                x_distance = vbx1 - obj_center_x
                edge_name = "좌측"
            # 객체가 차량의 오른쪽에 있는 경우
            elif obj_center_x > vbx2:
                x_distance = obj_center_x - vbx2
                edge_name = "우측"
            # 객체가 차량의 x 범위 내에 있는 경우
            else:
                x_distance = 0
                edge_name = "상하"

            # y 방향 거리 계산
            if obj_center_y < vby1:
                y_distance = vby1 - obj_center_y
            elif obj_center_y > vby2:
                y_distance = obj_center_y - vby2
            else:
                y_distance = 0

            # 최종 거리 계산 (x, y 중 하나라도 0이면 다른 하나만 사용)
            if x_distance == 0:
                distance = y_distance
            elif y_distance == 0:
                distance = x_distance
            else:
                # 둘 다 0이 아니면 유클리드 거리 계산
                distance = math.sqrt(x_distance ** 2 + y_distance ** 2)

            # 디버깅 로그
            logger.debug(
                f"[{self.name()}] 차량 경계({vbx1},{vby1},{vbx2},{vby2})와 객체 중심({obj_center_x},{obj_center_y}) 간 {edge_name} 거리: {distance:.1f}px")

            # 최소 거리 업데이트
            if distance < min_distance:
                min_distance = distance
                closest_edge = edge_name

        # 가장 가까운 차량과의 거리
        if min_distance <= config.distance_trash:
            logger.debug(
                f"[{self.name()}] 차량 {closest_edge} 경계와 충분히 가까움 (거리: {min_distance:.1f}px, 기준: {config.distance_trash}px) - 통과")
            return True
        else:
            logger.debug(
                f"[{self.name()}] 차량과 너무 멂 (거리: {min_distance:.1f}px, 기준: {config.distance_trash}px) - 배제")
            return False


class GravityDirectionStrategy(DetectionStrategy):
    """중력 방향 이동 확인 전략 - 연속적인 하강 움직임 감지"""

    def name(self):
        return "중력 방향 확인"

    def description(self):
        return "오브젝트가 중력 방향(아래쪽)으로 연속해서 이동하는지 확인"

    def check(self, frame, tracking_info, config, vehicle_info=None):
        logger.debug(f"[{self.name()}] 전략 검사 시작: tracking_info={len(tracking_info) if tracking_info else 0}")

        # 최소 프레임 수 확인 (최소 2개 필요)
        if len(tracking_info) < 2:
            logger.debug(f"[{self.name()}] 추적 정보 부족 (최소 2개 필요) - 배제")
            return False

        # 임계값 프레임 수까지 도달하지 않았으면 현재까지의 프레임으로 판단
        use_frames = min(len(tracking_info), config.min_frame_count_for_violation)

        # 현재 궤적에서 하강 움직임 확인 (최근 use_frames 개수만큼)
        y_diffs = []
        for i in range(1, len(tracking_info)):
            prev_pos = tracking_info[i - 1]['center']
            curr_pos = tracking_info[i]['center']
            y_diff = curr_pos[1] - prev_pos[1]
            y_diffs.append(y_diff)

        # 최근 프레임들의 하강 움직임 확인
        recent_diffs = y_diffs[-use_frames + 1:] if len(y_diffs) > use_frames - 1 else y_diffs

        # 각 프레임마다 최소 1픽셀 이상 아래로 이동했는지 확인
        downward_count = sum(1 for diff in recent_diffs if diff >= 1)

        # 대부분(80% 이상)의 프레임이 하강 움직임을 보이는지 확인
        threshold_count = max(1, int(len(recent_diffs) * 0.8))  # 최소 1개 이상 필요
        result = downward_count >= threshold_count

        if result:
            logger.debug(f"[{self.name()}] 연속적인 하강 움직임 감지 (하강 프레임: {downward_count}/{len(recent_diffs)}) - 통과")
        else:
            logger.debug(f"[{self.name()}] 연속적인 하강 움직임 부족 (하강 프레임: {downward_count}/{len(recent_diffs)}) - 배제")

        return result


class DirectionAlignmentStrategy(DetectionStrategy):
    """이동 방향과 차량 위치 정렬 확인 전략"""

    def name(self):
        return "이동방향 정렬 확인"

    def description(self):
        return "이동방향이 좌측인 경우 오브젝트가 바운딩박스 좌측과, 우측인 경우 바운딩박스 우측과 더 가까운지 확인"

    def check(self, frame, tracking_info, config, vehicle_info=None):
        logger.debug(f"[{self.name()}] 전략 검사 시작: tracking_info={len(tracking_info) if tracking_info else 0}")

        if len(tracking_info) < 2 or not tracking_info[-1].get('bbox') or not vehicle_info:
            logger.debug(f"[{self.name()}] 추적 정보 부족 또는 차량 정보 없음 - 통과")
            return True  # 정보 부족 시 통과 처리

        # 이전/현재 위치로 이동 방향 결정
        prev_pos = tracking_info[-2]['center']
        curr_pos = tracking_info[-1]['center']

        # x 좌표 변화로 좌/우 이동 확인
        x_diff = curr_pos[0] - prev_pos[0]

        # 객체 중심점
        obj_center_x = curr_pos[0]

        # 가장 가까운 차량 찾기
        nearest_vehicle = None
        min_distance = float('inf')

        for vehicle in vehicle_info:
            veh_bbox = vehicle['bbox']  # (x1, y1, x2, y2)
            veh_center_x = (veh_bbox[0] + veh_bbox[2]) / 2
            distance = abs(obj_center_x - veh_center_x)

            if distance < min_distance:
                min_distance = distance
                nearest_vehicle = vehicle

        if not nearest_vehicle:
            logger.debug(f"[{self.name()}] 근처에 차량 없음 - 통과")
            return True

        # 차량 바운딩 박스 중심점
        veh_bbox = nearest_vehicle['bbox']
        veh_center_x = (veh_bbox[0] + veh_bbox[2]) / 2

        # 객체가 차량의 우측에 있는지 확인
        obj_is_right_of_vehicle = obj_center_x > veh_center_x

        # 방향 결정 (x_diff가 양수면 우측 이동, 음수면 좌측 이동)
        if x_diff > config.horizontal_direction_threshold:
            # 우측 이동
            direction = "right"
            # 객체가 차량 우측에 있으면 통과
            result = obj_is_right_of_vehicle
            logger.debug(f"[{self.name()}] 우측 이동 감지 (x변화: {x_diff:.1f}px), "
                         f"객체는 차량의 {'우측' if obj_is_right_of_vehicle else '좌측'}에 위치 - "
                         f"{'통과' if result else '배제'}")
        elif x_diff < -config.horizontal_direction_threshold:
            # 좌측 이동
            direction = "left"
            # 객체가 차량 좌측에 있으면 통과
            result = not obj_is_right_of_vehicle
            logger.debug(f"[{self.name()}] 좌측 이동 감지 (x변화: {x_diff:.1f}px), "
                         f"객체는 차량의 {'우측' if obj_is_right_of_vehicle else '좌측'}에 위치 - "
                         f"{'통과' if result else '배제'}")
        else:
            # 수평 이동이 크지 않으면 통과
            logger.debug(f"[{self.name()}] 수평 이동 불명확 (x변화: {x_diff:.1f}px, "
                         f"기준: {config.horizontal_direction_threshold}px) - 통과")
            return True

        return result


class DetectionStrategyManager:
    """감지 전략을 관리하는 클래스"""

    def __init__(self):
        self.strategies = {}  # 전략 저장 딕셔너리
        self.enabled_strategies = set()  # 활성화된 전략 ID 집합
        logger.info("DetectionStrategyManager 초기화 완료")

    def register_strategy(self, strategy_id, strategy):
        """새로운 감지 전략 등록"""
        self.strategies[strategy_id] = strategy
        logger.info(f"전략 등록: {strategy_id} ({strategy.name()})")

    def enable_strategy(self, strategy_id):
        """전략 활성화"""
        if strategy_id in self.strategies:
            self.enabled_strategies.add(strategy_id)
            logger.info(f"전략 활성화: {strategy_id} ({self.strategies[strategy_id].name()})")
            return True
        logger.warning(f"존재하지 않는 전략 활성화 시도: {strategy_id}")
        return False

    def disable_strategy(self, strategy_id):
        """전략 비활성화"""
        if strategy_id in self.enabled_strategies:
            self.enabled_strategies.remove(strategy_id)
            logger.info(
                f"전략 비활성화: {strategy_id} ({self.strategies[strategy_id].name() if strategy_id in self.strategies else 'unknown'})")
            return True
        logger.warning(f"활성화되지 않은 전략을 비활성화 시도: {strategy_id}")
        return False

    def is_strategy_enabled(self, strategy_id):
        """전략이 활성화 되어있는지 확인"""
        return strategy_id in self.enabled_strategies

    def get_all_strategies(self):
        """모든 전략 반환"""
        return self.strategies

    def get_enabled_strategies(self):
        """활성화된 전략만 반환"""
        return {id: self.strategies[id] for id in self.enabled_strategies if id in self.strategies}

    def check_strategies(self, frame, tracking_info, config, vehicle_info=None):
        """
        모든 활성화된 전략 확인

        Args:
            frame: 현재 프레임
            tracking_info: 객체 추적 정보
            config: 전역 설정
            vehicle_info: 차량 정보

        Returns:
            dictionary: 각 전략 ID별 결과 (True/False)
        """
        logger.debug(f"전략 검사 시작: 활성화된 전략 수={len(self.enabled_strategies)}")

        # 카테고리별 전략 (추가/수정 가능)
        required_strategies = ['size_range', 'gravity_direction']  # 필수 전략
        optional_strategies = ['vehicle_distance', 'direction_alignment', 'vehicle_overlap']  # 선택적 전략

        results = {}

        # 각 전략 실행
        for strategy_id in self.enabled_strategies:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                try:
                    result = strategy.check(frame, tracking_info, config, vehicle_info)
                    results[strategy_id] = result
                    logger.debug(f"전략 '{strategy_id}' ({strategy.name()}) 결과: {result}")
                except Exception as e:
                    logger.error(f"전략 '{strategy_id}' ({strategy.name()}) 실행 중 오류: {str(e)}")
                    logger.error(traceback.format_exc())
                    results[strategy_id] = False  # 오류 발생 시 기본값 False
            else:
                logger.warning(f"존재하지 않는 전략 ID: {strategy_id}")

        # 결과 집계
        if config.detection_logic == "ANY":
            # 하나라도 True면 성공
            final_result = any(results.values()) if results else False
        elif config.detection_logic == "ALL":
            # 모두 True면 성공
            final_result = all(results.values()) if results else False
        elif config.detection_logic == "SMART":
            # 필수 전략은 모두 충족해야 하고, 선택적 전략은 하나 이상 충족해야 함
            required_results = [results.get(strategy_id, False) for strategy_id in required_strategies
                                if strategy_id in self.enabled_strategies]

            optional_results = [results.get(strategy_id, False) for strategy_id in optional_strategies
                                if strategy_id in self.enabled_strategies]

            required_pass = all(required_results) if required_results else False
            optional_pass = any(optional_results) if optional_results else True  # 활성화된 optional 전략이 없으면 통과

            final_result = required_pass and optional_pass
        else:
            # 기본값: ALL
            final_result = all(results.values()) if results else False

        # 모든 전략 검사 결과 로깅
        logger.debug(f"전체 전략 검사 결과: {final_result}, 상세={results}")

        return results

class VehicleAssociationStrategy(DetectionStrategy):
    """차량-쓰레기 객체 연관성 확인 전략"""

    def name(self):
        return "차량 연관성 확인"

    def description(self):
        return "쓰레기 객체가 특정 차량과 연관되어 있는지 확인"

    def check(self, frame, tracking_info, config, vehicle_info=None):
        logger.debug(
            f"[{self.name()}] 전략 검사 시작: tracking_info={len(tracking_info) if tracking_info else 0}, vehicle_info={len(vehicle_info) if vehicle_info else 0}")

        if not vehicle_info or not tracking_info:
            logger.debug(f"[{self.name()}] 차량 정보 또는 추적 정보 없음 - 배제")
            return False

        # 가장 최근 위치
        last_pos = tracking_info[-1]['center']

        # 각 차량과의 거리 계산
        distances = []
        for vehicle in vehicle_info:
            veh_center = vehicle['center']
            dist = math.sqrt((last_pos[0] - veh_center[0]) ** 2 + (last_pos[1] - veh_center[1]) ** 2)
            distances.append(dist)

        # 가장 가까운 차량과의 거리
        if distances:
            min_distance = min(distances)
            result = min_distance <= config.distance_trash

            if result:
                logger.debug(
                    f"[{self.name()}] 차량과 충분히 가까움 (거리: {min_distance:.1f}px, 기준: {config.distance_trash}px) - 통과")
            else:
                logger.debug(
                    f"[{self.name()}] 차량과 너무 멂 (거리: {min_distance:.1f}px, 기준: {config.distance_trash}px) - 배제")

            return result
        else:
            logger.debug(f"[{self.name()}] 유효한 차량 없음 - 배제")
            return False