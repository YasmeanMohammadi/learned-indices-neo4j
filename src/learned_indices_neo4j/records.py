from dataclasses import dataclass
from typing import Iterable


NumericValue = int | float


@dataclass(frozen=True, order=True)
class PropertyRecord:
    value: NumericValue
    node_id: str
    position: int


def normalize_numeric_value(value: object) -> NumericValue:
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid numeric index keys.")
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Empty strings are not valid numeric index keys.")
        return float(stripped) if "." in stripped else int(stripped)
    raise TypeError(f"Unsupported numeric index key: {value!r}")


def preprocess_pairs(pairs: Iterable[tuple[object, object]]) -> list[PropertyRecord]:
    cleaned = [
        (normalize_numeric_value(value), str(node_id))
        for value, node_id in pairs
        if value is not None and node_id is not None
    ]
    cleaned.sort(key=lambda item: (item[0], item[1]))

    return [
        PropertyRecord(value=value, node_id=node_id, position=position)
        for position, (value, node_id) in enumerate(cleaned)
    ]
