import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable

from learned_indices_neo4j.btree_index import BTreeIndex
from learned_indices_neo4j.io import read_records_csv
from learned_indices_neo4j.records import NumericValue, PropertyRecord
from learned_indices_neo4j.rmi_index import RMIIndex


@dataclass(frozen=True)
class PointQuery:
    property_name: str
    value: NumericValue
    node_id: str
    true_position: int


@dataclass(frozen=True)
class SplitRecords:
    train: list[PropertyRecord]
    test: list[PropertyRecord]


@dataclass(frozen=True)
class RMIGridResult:
    property_name: str
    k: int
    delta: int
    validation_mae: float
    validation_elements_examined_avg: float
    validation_coverage: float
    validation_miss_count: int


@dataclass(frozen=True)
class TunedRMIConfig:
    k: int
    delta: int


def split_records(
    records: list[PropertyRecord],
    *,
    train_fraction: float,
    seed: int,
) -> SplitRecords:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * train_fraction)
    return SplitRecords(train=shuffled[:split_at], test=shuffled[split_at:])


def sample_point_queries(
    property_name: str,
    records: list[PropertyRecord],
    *,
    query_count: int,
    seed: int,
) -> list[PointQuery]:
    if query_count < 1:
        raise ValueError("query_count must be positive.")

    sample_size = min(query_count, len(records))
    sampled = random.Random(seed).sample(records, sample_size)
    return [
        PointQuery(
            property_name=property_name,
            value=record.value,
            node_id=record.node_id,
            true_position=record.position,
        )
        for record in sampled
    ]


def write_point_queries(path: str | Path, queries: Iterable[PointQuery]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["property", "value", "node_id", "true_position"],
        )
        writer.writeheader()
        for query in queries:
            writer.writerow(
                {
                    "property": query.property_name,
                    "value": query.value,
                    "node_id": query.node_id,
                    "true_position": query.true_position,
                }
            )


def read_point_queries(path: str | Path) -> list[PointQuery]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            PointQuery(
                property_name=row["property"],
                value=_parse_numeric(row["value"]),
                node_id=row["node_id"],
                true_position=int(row["true_position"]),
            )
            for row in reader
        ]


def generate_workloads(
    *,
    data_dir: Path,
    workload_dir: Path,
    properties: list[str],
    train_fraction: float,
    query_count: int,
    seed: int,
) -> dict[str, Path]:
    workload_paths: dict[str, Path] = {}
    for offset, property_name in enumerate(properties):
        records = read_records_csv(data_dir / f"{property_name}.csv")
        split = split_records(records, train_fraction=train_fraction, seed=seed)
        queries = sample_point_queries(
            property_name,
            split.test,
            query_count=query_count,
            seed=seed + offset + 1,
        )
        output_path = workload_dir / f"{property_name}_point_queries.csv"
        write_point_queries(output_path, queries)
        workload_paths[property_name] = output_path
    return workload_paths


def tune_rmi_grid(
    *,
    property_name: str,
    records: list[PropertyRecord],
    train_records: list[PropertyRecord],
    k_candidates: list[int],
    delta_candidates: list[int],
    folds: int,
) -> tuple[TunedRMIConfig, list[RMIGridResult]]:
    if folds < 2:
        raise ValueError("RMI tuning requires at least two folds.")
    if not delta_candidates:
        raise ValueError("At least one delta candidate is required.")

    folds = min(folds, len(train_records))
    results: list[RMIGridResult] = []

    for k in k_candidates:
        if k < 1:
            continue
        for delta in delta_candidates:
            if delta < 0:
                continue

            errors: list[int] = []
            elements_examined: list[int] = []
            covered = 0
            total = 0

            for fold in range(folds):
                model_train = [
                    record
                    for index, record in enumerate(train_records)
                    if index % folds != fold
                ]
                validation = [
                    record
                    for index, record in enumerate(train_records)
                    if index % folds == fold
                ]
                if not model_train or not validation:
                    continue

                index = RMIIndex(records, k=k, delta=delta, training_records=model_train)
                for record in validation:
                    error = index.prediction_error(record)
                    errors.append(error)
                    elements_examined.append(index.elements_examined(record.value))
                    covered += int(index.covers_position(record))
                    total += 1

            coverage = covered / total if total else 0.0
            results.append(
                RMIGridResult(
                    property_name=property_name,
                    k=k,
                    delta=delta,
                    validation_mae=mean(errors) if errors else 0.0,
                    validation_elements_examined_avg=mean(elements_examined)
                    if elements_examined
                    else 0.0,
                    validation_coverage=coverage,
                    validation_miss_count=total - covered,
                )
            )

    if not results:
        raise ValueError("No valid RMI tuning configurations were evaluated.")

    best = min(
        results,
        key=lambda result: (
            -result.validation_coverage,
            result.validation_elements_examined_avg,
            result.validation_mae,
            result.k,
            result.delta,
        ),
    )
    return TunedRMIConfig(k=best.k, delta=best.delta), results


def run_experiments(
    *,
    data_dir: Path,
    workload_dir: Path,
    results_dir: Path,
    properties: list[str],
    train_fraction: float,
    query_count: int,
    seed: int,
    btree_order: int,
    k_candidates: list[int],
    delta_candidates: list[int],
    folds: int,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    workload_dir.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    worst_case_rows: list[dict[str, object]] = []

    for offset, property_name in enumerate(properties):
        records = read_records_csv(data_dir / f"{property_name}.csv")
        split = split_records(records, train_fraction=train_fraction, seed=seed)
        workload_path = workload_dir / f"{property_name}_point_queries.csv"
        if workload_path.exists():
            queries = read_point_queries(workload_path)
        else:
            queries = sample_point_queries(
                property_name,
                split.test,
                query_count=query_count,
                seed=seed + offset + 1,
            )
            write_point_queries(workload_path, queries)

        tuned, grid_results = tune_rmi_grid(
            property_name=property_name,
            records=records,
            train_records=split.train,
            k_candidates=k_candidates,
            delta_candidates=delta_candidates,
            folds=folds,
        )
        sensitivity_rows.extend(_grid_result_to_row(result) for result in grid_results)

        btree_metrics = _evaluate_btree(records, queries, order=btree_order)
        rmi_metrics, worst_cases = _evaluate_rmi(
            records,
            split.train,
            queries,
            property_name=property_name,
            k=tuned.k,
            delta=tuned.delta,
        )
        worst_case_rows.extend(worst_cases)

        main_rows.extend(
            _main_metric_rows(
                property_name=property_name,
                btree_metrics=btree_metrics,
                rmi_metrics=rmi_metrics,
            )
        )

    output_paths = {
        "main_comparison": results_dir / "main_comparison.csv",
        "rmi_hyperparameter_sensitivity": results_dir
        / "rmi_hyperparameter_sensitivity.csv",
        "rmi_worst_case_queries": results_dir / "rmi_worst_case_queries.csv",
    }
    _write_dict_rows(output_paths["main_comparison"], main_rows)
    _write_dict_rows(output_paths["rmi_hyperparameter_sensitivity"], sensitivity_rows)
    _write_dict_rows(output_paths["rmi_worst_case_queries"], worst_case_rows)
    return output_paths


def _evaluate_btree(
    records: list[PropertyRecord],
    queries: list[PointQuery],
    *,
    order: int,
) -> dict[str, float]:
    evaluation_started = perf_counter()
    build_started = perf_counter()
    index = BTreeIndex(records, order=order)
    build_time_ms = (perf_counter() - build_started) * 1000
    comparisons = math.ceil(math.log2(len(records))) if len(records) > 1 else len(records)
    latencies: list[float] = []

    for query in queries:
        started = perf_counter()
        index.exact(query.value)
        latencies.append((perf_counter() - started) * 1000)

    return {
        "mae": math.nan,
        "elements_examined_avg": float(comparisons),
        "build_time_ms": build_time_ms,
        "execution_time_ms": (perf_counter() - evaluation_started) * 1000,
        "latency_avg_ms": mean(latencies) if latencies else 0.0,
        "latency_min_ms": min(latencies) if latencies else 0.0,
        "latency_max_ms": max(latencies) if latencies else 0.0,
    }


def _evaluate_rmi(
    records: list[PropertyRecord],
    train_records: list[PropertyRecord],
    queries: list[PointQuery],
    *,
    property_name: str,
    k: int,
    delta: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    evaluation_started = perf_counter()
    build_started = perf_counter()
    index = RMIIndex(records, k=k, delta=delta, training_records=train_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    errors: list[int] = []
    elements_examined: list[int] = []
    covered = 0
    worst_cases: list[dict[str, object]] = []

    for query in queries:
        sampled_record = PropertyRecord(
            value=query.value,
            node_id=query.node_id,
            position=query.true_position,
        )
        started = perf_counter()
        predicted_position = index.predict_position(query.value)
        start, end = index.prediction_window(query.value)
        for record in index.records[start:end]:
            if record.position == query.true_position:
                break
        latencies.append((perf_counter() - started) * 1000)

        error = abs(predicted_position - query.true_position)
        is_covered = index.covers_position(sampled_record)
        errors.append(error)
        elements_examined.append(end - start)
        covered += int(is_covered)
        worst_cases.append(
            {
                "property": property_name,
                "value": query.value,
                "node_id": query.node_id,
                "true_position": query.true_position,
                "predicted_position": predicted_position,
                "absolute_error": error,
                "covered_by_delta": is_covered,
                "k": k,
                "delta": delta,
                "window_start": start,
                "window_end": end,
            }
        )

    worst_cases.sort(key=lambda row: int(row["absolute_error"]), reverse=True)

    return (
        {
            "mae": mean(errors) if errors else 0.0,
            "elements_examined_avg": mean(elements_examined) if elements_examined else 0.0,
            "build_time_ms": build_time_ms,
            "execution_time_ms": (perf_counter() - evaluation_started) * 1000,
            "latency_avg_ms": mean(latencies) if latencies else 0.0,
            "latency_min_ms": min(latencies) if latencies else 0.0,
            "latency_max_ms": max(latencies) if latencies else 0.0,
            "coverage": covered / len(queries) if queries else 0.0,
            "k": float(k),
            "delta": float(delta),
        },
        worst_cases[:20],
    )


def _main_metric_rows(
    *,
    property_name: str,
    btree_metrics: dict[str, float],
    rmi_metrics: dict[str, float],
) -> list[dict[str, object]]:
    return [
        {
            "property": property_name,
            "metric": "MAE (positions)",
            "B-Tree": "",
            "RMI": _format_float(rmi_metrics["mae"]),
        },
        {
            "property": property_name,
            "metric": "Elements examined (avg)",
            "B-Tree": _format_float(btree_metrics["elements_examined_avg"]),
            "RMI": _format_float(rmi_metrics["elements_examined_avg"]),
        },
        {
            "property": property_name,
            "metric": "Index build time (ms)",
            "B-Tree": _format_float(btree_metrics["build_time_ms"]),
            "RMI": _format_float(rmi_metrics["build_time_ms"]),
        },
        {
            "property": property_name,
            "metric": "Evaluation execution time (ms)",
            "B-Tree": _format_float(btree_metrics["execution_time_ms"]),
            "RMI": _format_float(rmi_metrics["execution_time_ms"]),
        },
        {
            "property": property_name,
            "metric": "Query latency (ms, avg)",
            "B-Tree": _format_float(btree_metrics["latency_avg_ms"]),
            "RMI": _format_float(rmi_metrics["latency_avg_ms"]),
        },
        {
            "property": property_name,
            "metric": "Query latency (ms, min)",
            "B-Tree": _format_float(btree_metrics["latency_min_ms"]),
            "RMI": _format_float(rmi_metrics["latency_min_ms"]),
        },
        {
            "property": property_name,
            "metric": "Query latency (ms, max)",
            "B-Tree": _format_float(btree_metrics["latency_max_ms"]),
            "RMI": _format_float(rmi_metrics["latency_max_ms"]),
        },
        {
            "property": property_name,
            "metric": "RMI coverage",
            "B-Tree": "",
            "RMI": _format_float(rmi_metrics["coverage"]),
        },
        {
            "property": property_name,
            "metric": "RMI k",
            "B-Tree": "",
            "RMI": int(rmi_metrics["k"]),
        },
        {
            "property": property_name,
            "metric": "RMI delta",
            "B-Tree": "",
            "RMI": int(rmi_metrics["delta"]),
        },
    ]


def _grid_result_to_row(result: RMIGridResult) -> dict[str, object]:
    return {
        "property": result.property_name,
        "k": result.k,
        "delta": result.delta,
        "validation_mae": _format_float(result.validation_mae),
        "validation_elements_examined_avg": _format_float(
            result.validation_elements_examined_avg
        ),
        "validation_coverage": _format_float(result.validation_coverage),
        "validation_miss_count": result.validation_miss_count,
    }


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_numeric(value: str) -> NumericValue:
    return float(value) if "." in value else int(value)


def _format_float(value: float) -> str:
    return f"{value:.6f}"
