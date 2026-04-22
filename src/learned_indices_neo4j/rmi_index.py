from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from learned_indices_neo4j.records import NumericValue, PropertyRecord


@dataclass(frozen=True)
class LinearModel:
    slope: float
    intercept: float

    @classmethod
    def fit(cls, records: Iterable[PropertyRecord]) -> "LinearModel":
        points = [(float(record.value), float(record.position)) for record in records]
        if not points:
            return cls(slope=0.0, intercept=0.0)

        x_mean = mean(x for x, _ in points)
        y_mean = mean(y for _, y in points)
        denominator = sum((x - x_mean) ** 2 for x, _ in points)
        if denominator == 0:
            return cls(slope=0.0, intercept=y_mean)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        return cls(slope=slope, intercept=intercept)

    def predict(self, value: NumericValue) -> float:
        return self.slope * float(value) + self.intercept


@dataclass(frozen=True)
class RMIModels:
    stage_1: LinearModel
    stage_2: list[LinearModel]
    k: int
    total_records: int


@dataclass(frozen=True)
class RMITuningResult:
    k: int
    delta: int
    mean_abs_error: float
    max_abs_error: int
    coverage: float


class RMIIndex:
    """Two-stage recursive model index with bounded local scans."""

    def __init__(
        self,
        records: Iterable[PropertyRecord],
        *,
        k: int = 16,
        delta: int | None = None,
    ) -> None:
        if k < 1:
            raise ValueError("RMI k must be at least 1.")

        self.records = self._normalize_records(records)
        self.k = k
        self.models = self._train_models(self.records, k=k, total_records=len(self.records))
        self.delta = self._calibrate_delta() if delta is None else delta
        if self.delta < 0:
            raise ValueError("RMI delta must be non-negative.")

    def __len__(self) -> int:
        return len(self.records)

    def exact(self, value: NumericValue) -> list[PropertyRecord]:
        if not self.records:
            return []

        start, end = self._prediction_window(value)
        matches = [record for record in self.records[start:end] if record.value == value]
        if not matches:
            return []

        first = matches[0].position
        last = matches[-1].position
        while first > 0 and self.records[first - 1].value == value:
            first -= 1
        while last + 1 < len(self.records) and self.records[last + 1].value == value:
            last += 1

        return self.records[first : last + 1]

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
        matches: list[PropertyRecord] = []

        for record in self.records[start:]:
            if maximum is not None:
                if include_maximum and record.value > maximum:
                    break
                if not include_maximum and record.value >= maximum:
                    break
            matches.append(record)

        return matches

    def predict_position(self, value: NumericValue) -> int:
        return self._predict_position_with_models(self.models, value)

    def prediction_error(self, record: PropertyRecord) -> int:
        return abs(self.predict_position(record.value) - record.position)

    def _prediction_window(self, value: NumericValue) -> tuple[int, int]:
        predicted_position = self.predict_position(value)
        start = max(0, predicted_position - self.delta)
        end = min(len(self.records), predicted_position + self.delta + 1)
        return start, end

    def _lower_bound(
        self,
        value: NumericValue | None,
        *,
        include_minimum: bool,
    ) -> int:
        if value is None:
            return 0

        start, end = self._prediction_window(value)
        candidate = None
        for index in range(start, end):
            record_value = self.records[index].value
            if include_minimum and record_value >= value:
                candidate = index
                break
            if not include_minimum and record_value > value:
                candidate = index
                break

        if candidate is None:
            candidate = end
            while candidate < len(self.records):
                record_value = self.records[candidate].value
                if include_minimum and record_value >= value:
                    break
                if not include_minimum and record_value > value:
                    break
                candidate += 1

        while candidate > 0:
            previous = self.records[candidate - 1].value
            if include_minimum and previous < value:
                break
            if not include_minimum and previous <= value:
                break
            candidate -= 1

        return candidate

    def _calibrate_delta(self) -> int:
        if not self.records:
            return 0
        return max(self.prediction_error(record) for record in self.records)

    @staticmethod
    def _normalize_records(records: Iterable[PropertyRecord]) -> list[PropertyRecord]:
        ordered = sorted(records, key=lambda record: (record.value, record.node_id))
        return [
            PropertyRecord(value=record.value, node_id=record.node_id, position=position)
            for position, record in enumerate(ordered)
        ]

    @classmethod
    def _train_models(
        cls,
        records: list[PropertyRecord],
        *,
        k: int,
        total_records: int,
    ) -> RMIModels:
        stage_1 = LinearModel.fit(records)
        partitions: list[list[PropertyRecord]] = [[] for _ in range(k)]
        shell = RMIModels(stage_1=stage_1, stage_2=[], k=k, total_records=total_records)

        for record in records:
            partitions[cls._partition_for_value(shell, record.value)].append(record)

        stage_2 = [
            LinearModel.fit(partition) if partition else stage_1
            for partition in partitions
        ]
        return RMIModels(stage_1=stage_1, stage_2=stage_2, k=k, total_records=total_records)

    @classmethod
    def _partition_for_value(cls, models: RMIModels, value: NumericValue) -> int:
        if models.k == 1 or models.total_records <= 1:
            return 0

        predicted_position = models.stage_1.predict(value)
        scaled = predicted_position / max(1, models.total_records - 1)
        partition = int(scaled * models.k)
        return min(max(partition, 0), models.k - 1)

    @classmethod
    def _predict_position_with_models(cls, models: RMIModels, value: NumericValue) -> int:
        if models.total_records == 0:
            return 0

        partition = cls._partition_for_value(models, value)
        predicted = round(models.stage_2[partition].predict(value))
        return min(max(predicted, 0), models.total_records - 1)


def tune_rmi(
    records: Iterable[PropertyRecord],
    *,
    k_candidates: Iterable[int] = (4, 8, 16, 32),
    delta_candidates: Iterable[int] | None = None,
    folds: int = 5,
) -> RMITuningResult:
    normalized = RMIIndex._normalize_records(records)
    if not normalized:
        return RMITuningResult(k=1, delta=0, mean_abs_error=0.0, max_abs_error=0, coverage=1.0)

    if folds < 2:
        raise ValueError("RMI cross-validation requires at least two folds.")

    folds = min(folds, len(normalized))
    candidates = [candidate for candidate in k_candidates if candidate >= 1]
    if not candidates:
        raise ValueError("At least one positive k candidate is required.")

    explicit_deltas = None
    if delta_candidates is not None:
        explicit_deltas = sorted(delta for delta in delta_candidates if delta >= 0)
        if not explicit_deltas:
            raise ValueError("At least one non-negative delta candidate is required.")

    best: RMITuningResult | None = None
    best_sort_key: tuple[float, float, int, int] | None = None

    for k in candidates:
        errors: list[int] = []
        for fold in range(folds):
            training = [record for index, record in enumerate(normalized) if index % folds != fold]
            validation = [record for index, record in enumerate(normalized) if index % folds == fold]
            if not training or not validation:
                continue

            models = RMIIndex._train_models(training, k=k, total_records=len(normalized))
            errors.extend(
                abs(RMIIndex._predict_position_with_models(models, record.value) - record.position)
                for record in validation
            )

        if not errors:
            continue

        mean_abs_error = mean(errors)
        max_abs_error = max(errors)
        if explicit_deltas is None:
            delta = max_abs_error
            coverage = 1.0
        else:
            coverage_by_delta = [
                (sum(error <= delta for error in errors) / len(errors), delta)
                for delta in explicit_deltas
            ]
            coverage, delta = max(coverage_by_delta, key=lambda item: (item[0], -item[1]))

        result = RMITuningResult(
            k=k,
            delta=delta,
            mean_abs_error=mean_abs_error,
            max_abs_error=max_abs_error,
            coverage=coverage,
        )
        sort_key = (-coverage, mean_abs_error, delta, k)
        if best is None or sort_key < best_sort_key:
            best = result
            best_sort_key = sort_key

    if best is None:
        raise ValueError("Unable to tune RMI with the provided data and folds.")

    return best
