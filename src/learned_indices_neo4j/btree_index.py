from dataclasses import dataclass, field
from typing import Iterable

from learned_indices_neo4j.records import NumericValue, PropertyRecord, preprocess_pairs


@dataclass
class BTreeNode:
    keys: list[NumericValue]
    children: list["BTreeNode"] = field(default_factory=list)
    records: list[PropertyRecord] = field(default_factory=list)
    next_leaf: "BTreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return not self.children


class BTreeIndex:
    """Local B+ tree index for exact and range lookups over property records."""

    def __init__(self, records: Iterable[PropertyRecord], order: int = 64) -> None:
        if order < 3:
            raise ValueError("B-tree order must be at least 3.")

        self.order = order
        self.records = self._normalize_records(records)
        self.root = self._bulk_load(self.records)

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[object, object]],
        *,
        order: int = 64,
    ) -> "BTreeIndex":
        return cls(preprocess_pairs(pairs), order=order)

    def __len__(self) -> int:
        return len(self.records)

    def exact(self, value: NumericValue) -> list[PropertyRecord]:
        return self.range(value, value)

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

        leaf = self._find_leaf(minimum) if minimum is not None else self._leftmost_leaf()
        matches: list[PropertyRecord] = []

        while leaf is not None:
            for record in leaf.records:
                if minimum is not None:
                    if include_minimum and record.value < minimum:
                        continue
                    if not include_minimum and record.value <= minimum:
                        continue

                if maximum is not None:
                    if include_maximum and record.value > maximum:
                        return matches
                    if not include_maximum and record.value >= maximum:
                        return matches

                matches.append(record)

            leaf = leaf.next_leaf

        return matches

    def height(self) -> int:
        height = 0
        node = self.root
        while node is not None:
            height += 1
            node = node.children[0] if node.children else None
        return height

    @staticmethod
    def _normalize_records(records: Iterable[PropertyRecord]) -> list[PropertyRecord]:
        ordered = sorted(records, key=lambda record: (record.value, record.node_id))
        return [
            PropertyRecord(value=record.value, node_id=record.node_id, position=position)
            for position, record in enumerate(ordered)
        ]

    def _bulk_load(self, records: list[PropertyRecord]) -> BTreeNode:
        if not records:
            return BTreeNode(keys=[])

        leaves: list[BTreeNode] = []
        previous_leaf: BTreeNode | None = None
        for start in range(0, len(records), self.order):
            leaf_records = records[start : start + self.order]
            leaf = BTreeNode(keys=[record.value for record in leaf_records], records=leaf_records)
            if previous_leaf is not None:
                previous_leaf.next_leaf = leaf
            leaves.append(leaf)
            previous_leaf = leaf

        level = leaves
        while len(level) > 1:
            next_level: list[BTreeNode] = []
            for start in range(0, len(level), self.order):
                children = level[start : start + self.order]
                separator_keys = [self._first_key(child) for child in children[1:]]
                next_level.append(BTreeNode(keys=separator_keys, children=children))
            level = next_level

        return level[0]

    def _find_leaf(self, value: NumericValue | None) -> BTreeNode:
        node = self.root
        while not node.is_leaf:
            child_index = 0
            while child_index < len(node.keys) and value is not None and value > node.keys[child_index]:
                child_index += 1
            node = node.children[child_index]
        return node

    def _leftmost_leaf(self) -> BTreeNode:
        node = self.root
        while not node.is_leaf:
            node = node.children[0]
        return node

    @staticmethod
    def _first_key(node: BTreeNode) -> NumericValue:
        while not node.is_leaf:
            node = node.children[0]
        return node.records[0].value
