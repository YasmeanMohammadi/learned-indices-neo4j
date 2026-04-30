from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Iterable

from learned_indices_neo4j.records import NumericValue, PropertyRecord, preprocess_pairs


@dataclass(frozen=True)
class PGMSegment:
    start_key: NumericValue
    slope: float
    intercept: float
    start_index: int
    end_index: int

    def predict(self, value: NumericValue) -> int:
        return round(self.slope * float(value) + self.intercept)


@dataclass(frozen=True)
class PGMLevel:
    segments: list[PGMSegment]
    start_keys: list[NumericValue]


class StaticPGMIndex:
    """Static recursive PGM-index with epsilon-bounded binary refinement."""

    def __init__(
        self,
        records: Iterable[PropertyRecord],
        *,
        epsilon: int = 64,
        training_records: Iterable[PropertyRecord] | None = None,
    ) -> None:
        if epsilon < 1:
            raise ValueError("PGM epsilon must be at least 1.")

        self.epsilon = epsilon
        self.records = self._normalize_records(records)
        self._keys = [record.value for record in self.records]
        self.training_records = (
            self._sort_records_preserving_positions(training_records)
            if training_records is not None
            else self.records
        )
        self._training_points = self._distinct_points(self.training_records)
        self.levels = self._build_levels(self._training_points)

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[object, object]],
        *,
        epsilon: int = 64,
    ) -> "StaticPGMIndex":
        return cls(preprocess_pairs(pairs), epsilon=epsilon)

    def __len__(self) -> int:
        return len(self.records)

    def exact(self, value: NumericValue) -> list[PropertyRecord]:
        if not self.records:
            return []

        start, end = self.prediction_window(value)
        first = bisect_left(self._keys, value, start, end)
        if first >= end or self.records[first].value != value:
            return []

        last_exclusive = bisect_right(self._keys, value, first, end)
        while first > 0 and self.records[first - 1].value == value:
            first -= 1
        while last_exclusive < len(self.records) and self.records[last_exclusive].value == value:
            last_exclusive += 1
        return self.records[first:last_exclusive]

    def range(
        self,
        minimum: NumericValue | None = None,
        maximum: NumericValue | None = None,
        *,
        include_minimum: bool = True,
        include_maximum: bool = True,
    ) -> list[PropertyRecord]:
        if not self.records:
            return []

        start = self._lower_bound(minimum, include_minimum=include_minimum)
        if maximum is None:
            end = len(self.records)
        else:
            end = self._upper_bound(maximum, include_maximum=include_maximum)
        return self.records[start:end]

    def predict_position(self, value: NumericValue) -> int:
        segment = self._locate_bottom_segment(value)
        predicted = segment.predict(value)
        return min(max(predicted, 0), len(self.records) - 1 if self.records else 0)

    def prediction_window(self, value: NumericValue) -> tuple[int, int]:
        predicted = self.predict_position(value)
        start = max(0, predicted - self.epsilon)
        end = min(len(self.records), predicted + self.epsilon + 1)
        return start, end

    def elements_examined(self, value: NumericValue) -> int:
        return self.binary_search_comparisons(value)

    def binary_search_comparisons(self, value: NumericValue) -> int:
        route = 0
        if len(self.levels) > 1:
            window = min(len(self.levels[-2].segments), 2 * self.epsilon + 1)
            route = (len(self.levels) - 1) * math.ceil(math.log2(max(1, window)))

        start, end = self.prediction_window(value)
        leaf = math.ceil(math.log2(max(1, end - start)))
        first = bisect_left(self._keys, value, start, end)
        if first >= end or self.records[first].value != value:
            return route + leaf

        left_expansion = 0
        probe = first
        while probe > 0 and self.records[probe - 1].value == value:
            left_expansion += 1
            probe -= 1

        right_expansion = 0
        probe = bisect_right(self._keys, value, first, end)
        while probe < len(self.records) and self.records[probe].value == value:
            right_expansion += 1
            probe += 1

        return route + leaf + left_expansion + right_expansion

    def boundary_search_comparisons(self, value: NumericValue | None) -> int:
        if value is None or not self.records:
            return 0

        route = 0
        if len(self.levels) > 1:
            window = min(len(self.levels[-2].segments), 2 * self.epsilon + 1)
            route = (len(self.levels) - 1) * math.ceil(math.log2(max(1, window)))

        start, end = self.prediction_window(value)
        leaf = math.ceil(math.log2(max(1, end - start)))
        return route + leaf

    def covers_position(self, record: PropertyRecord) -> bool:
        start, end = self.prediction_window(record.value)
        return start <= record.position < end

    def prediction_error(self, record: PropertyRecord) -> int:
        return abs(self.predict_position(record.value) - record.position)

    def _lower_bound(
        self,
        value: NumericValue | None,
        *,
        include_minimum: bool,
    ) -> int:
        if value is None:
            return 0

        start, end = self.prediction_window(value)
        candidate = (
            bisect_left(self._keys, value, start, end)
            if include_minimum
            else bisect_right(self._keys, value, start, end)
        )
        while candidate > 0:
            previous = self.records[candidate - 1].value
            if include_minimum and previous < value:
                break
            if not include_minimum and previous <= value:
                break
            candidate -= 1
        while candidate < len(self.records):
            current = self.records[candidate].value
            if include_minimum and current >= value:
                break
            if not include_minimum and current > value:
                break
            candidate += 1
        return candidate

    def _upper_bound(
        self,
        value: NumericValue,
        *,
        include_maximum: bool,
    ) -> int:
        start, end = self.prediction_window(value)
        candidate = (
            bisect_right(self._keys, value, start, end)
            if include_maximum
            else bisect_left(self._keys, value, start, end)
        )
        while candidate < len(self.records):
            current = self.records[candidate].value
            if include_maximum and current > value:
                break
            if not include_maximum and current >= value:
                break
            candidate += 1
        return candidate

    @staticmethod
    def _normalize_records(records: Iterable[PropertyRecord]) -> list[PropertyRecord]:
        ordered = sorted(records, key=lambda record: (record.value, record.node_id))
        return [
            PropertyRecord(value=record.value, node_id=record.node_id, position=position)
            for position, record in enumerate(ordered)
        ]

    @staticmethod
    def _sort_records_preserving_positions(
        records: Iterable[PropertyRecord],
    ) -> list[PropertyRecord]:
        return sorted(records, key=lambda record: (record.value, record.node_id))

    @staticmethod
    def _distinct_points(records: list[PropertyRecord]) -> list[tuple[NumericValue, int]]:
        points: list[tuple[NumericValue, int]] = []
        last_value: NumericValue | None = None
        for record in records:
            if last_value is None or record.value != last_value:
                points.append((record.value, record.position))
                last_value = record.value
        return points

    def _build_levels(self, points: list[tuple[NumericValue, int]]) -> list[PGMLevel]:
        if not points:
            return [PGMLevel(segments=[], start_keys=[])]

        levels: list[PGMLevel] = []
        current_points = points
        while True:
            segments = self._build_segments(current_points)
            levels.append(PGMLevel(segments=segments, start_keys=[segment.start_key for segment in segments]))
            if len(segments) == 1:
                break
            current_points = [
                (segment.start_key, index)
                for index, segment in enumerate(segments)
            ]
        return levels

    def _build_segments(self, points: list[tuple[NumericValue, int]]) -> list[PGMSegment]:
        if len(points) == 1:
            key, position = points[0]
            return [
                PGMSegment(
                    start_key=key,
                    slope=0.0,
                    intercept=float(position),
                    start_index=0,
                    end_index=0,
                )
            ]

        segments: list[PGMSegment] = []
        start = 0
        while start < len(points):
            if start == len(points) - 1:
                key, position = points[start]
                segments.append(
                    PGMSegment(
                        start_key=key,
                        slope=0.0,
                        intercept=float(position),
                        start_index=start,
                        end_index=start,
                    )
                )
                break

            x0, y0 = points[start]
            lower_slope = float("-inf")
            upper_slope = float("inf")
            end = start

            for index in range(start + 1, len(points)):
                x, y = points[index]
                dx = float(x - x0)
                if dx == 0:
                    continue
                min_slope = (float(y - self.epsilon) - float(y0)) / dx
                max_slope = (float(y + self.epsilon) - float(y0)) / dx
                candidate_lower = max(lower_slope, min_slope)
                candidate_upper = min(upper_slope, max_slope)
                if candidate_lower <= candidate_upper:
                    lower_slope = candidate_lower
                    upper_slope = candidate_upper
                    end = index
                else:
                    break

            slope = 0.0 if lower_slope == float("-inf") else (lower_slope + upper_slope) / 2.0
            intercept = float(y0) - slope * float(x0)
            segments.append(
                PGMSegment(
                    start_key=x0,
                    slope=slope,
                    intercept=intercept,
                    start_index=start,
                    end_index=end,
                )
            )
            start = end + 1

        return segments

    def _locate_bottom_segment(self, value: NumericValue) -> PGMSegment:
        if not self.levels or not self.levels[-1].segments:
            raise ValueError("Cannot query an empty PGM-index.")

        segment = self.levels[-1].segments[0]
        for level_index in range(len(self.levels) - 2, -1, -1):
            lower = self.levels[level_index]
            predicted_child = min(
                max(segment.predict(value), 0),
                len(lower.segments) - 1,
            )
            left = max(0, predicted_child - self.epsilon)
            right = min(len(lower.segments), predicted_child + self.epsilon + 1)
            slice_keys = lower.start_keys[left:right]
            child_offset = bisect_right(slice_keys, value) - 1
            if child_offset < 0:
                child_offset = 0
            segment = lower.segments[left + child_offset]
        return segment
