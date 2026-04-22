import csv
from pathlib import Path
from typing import Iterable

from learned_indices_neo4j.records import PropertyRecord


def write_records_csv(path: str | Path, records: Iterable[PropertyRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["value", "node_id", "position"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "value": record.value,
                    "node_id": record.node_id,
                    "position": record.position,
                }
            )


def read_records_csv(path: str | Path) -> list[PropertyRecord]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            PropertyRecord(
                value=float(row["value"]) if "." in row["value"] else int(row["value"]),
                node_id=row["node_id"],
                position=int(row["position"]),
            )
            for row in reader
        ]
