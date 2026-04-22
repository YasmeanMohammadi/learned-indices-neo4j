from bisect import bisect_left, bisect_right
from typing import Iterable

from learned_indices_neo4j.records import NumericValue, PropertyRecord, preprocess_pairs


class SortedArrayIndex:
    """Baseline index using binary search over a sorted array."""

    def __init__(self, records: Iterable[PropertyRecord]) -> None:
        ordered = sorted(records, key=lambda record: (record.value, record.node_id))
        self.records = [
            PropertyRecord(value=record.value, node_id=record.node_id, position=position)
            for position, record in enumerate(ordered)
        ]
        self._keys = [record.value for record in self.records]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[object, object]]) -> "SortedArrayIndex":
        return cls(preprocess_pairs(pairs))

    def __len__(self) -> int:
        return len(self.records)

    def lower_bound(self, value: NumericValue) -> int:
        return bisect_left(self._keys, value)

    def upper_bound(self, value: NumericValue) -> int:
        return bisect_right(self._keys, value)

    def estimate_position(self, value: NumericValue) -> int:
        """Return the insertion position found by binary search."""
        return self.lower_bound(value)

    def exact(self, value: NumericValue) -> list[PropertyRecord]:
        start = self.lower_bound(value)
        end = self.upper_bound(value)
        return self.records[start:end]

    def range(
        self,
        minimum: NumericValue | None = None,
        maximum: NumericValue | None = None,
        *,
        include_minimum: bool = True,
        include_maximum: bool = True,
    ) -> list[PropertyRecord]:
        if minimum is None:
            start = 0
        elif include_minimum:
            start = self.lower_bound(minimum)
        else:
            start = self.upper_bound(minimum)

        if maximum is None:
            end = len(self.records)
        elif include_maximum:
            end = self.upper_bound(maximum)
        else:
            end = self.lower_bound(maximum)

        return self.records[start:end]
