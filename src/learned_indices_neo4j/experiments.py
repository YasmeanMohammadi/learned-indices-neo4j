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
from learned_indices_neo4j.pgm_index import StaticPGMIndex
from learned_indices_neo4j.records import NumericValue, PropertyRecord
from learned_indices_neo4j.rmi_index import RMIIndex


@dataclass(frozen=True)
class PointQuery:
    property_name: str
    value: NumericValue
    node_id: str
    true_position: int


@dataclass(frozen=True)
class RangeQuery:
    property_name: str
    minimum_value: NumericValue
    maximum_value: NumericValue
    true_start_position: int
    true_end_position: int
    true_result_count: int


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


@dataclass(frozen=True)
class PGMGridResult:
    property_name: str
    epsilon: int
    validation_mae: float
    validation_elements_examined_avg: float
    validation_coverage: float
    validation_miss_count: int


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


def sample_range_queries(
    property_name: str,
    sample_records: list[PropertyRecord],
    full_records: list[PropertyRecord],
    *,
    query_count: int,
    seed: int,
) -> list[RangeQuery]:
    if query_count < 1:
        raise ValueError("query_count must be positive.")
    if not full_records:
        return []

    source_records = sample_records or full_records
    randomizer = random.Random(seed)
    queries: list[RangeQuery] = []
    full_index = BTreeIndex(full_records, order=64)

    for _ in range(query_count):
        left = randomizer.choice(source_records)
        right = randomizer.choice(source_records)
        minimum_value = min(left.value, right.value)
        maximum_value = max(left.value, right.value)
        matches = full_index.range(minimum_value, maximum_value)
        if not matches:
            continue
        queries.append(
            RangeQuery(
                property_name=property_name,
                minimum_value=minimum_value,
                maximum_value=maximum_value,
                true_start_position=matches[0].position,
                true_end_position=matches[-1].position,
                true_result_count=len(matches),
            )
        )

    return queries


def write_range_queries(path: str | Path, queries: Iterable[RangeQuery]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "property",
                "minimum_value",
                "maximum_value",
                "true_start_position",
                "true_end_position",
                "true_result_count",
            ],
        )
        writer.writeheader()
        for query in queries:
            writer.writerow(
                {
                    "property": query.property_name,
                    "minimum_value": query.minimum_value,
                    "maximum_value": query.maximum_value,
                    "true_start_position": query.true_start_position,
                    "true_end_position": query.true_end_position,
                    "true_result_count": query.true_result_count,
                }
            )


def read_range_queries(path: str | Path) -> list[RangeQuery]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            RangeQuery(
                property_name=row["property"],
                minimum_value=_parse_numeric(row["minimum_value"]),
                maximum_value=_parse_numeric(row["maximum_value"]),
                true_start_position=int(row["true_start_position"]),
                true_end_position=int(row["true_end_position"]),
                true_result_count=int(row["true_result_count"]),
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


def generate_range_workloads(
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
        queries = sample_range_queries(
            property_name,
            split.test,
            records,
            query_count=query_count,
            seed=seed + offset + 101,
        )
        output_path = workload_dir / f"{property_name}_range_queries.csv"
        write_range_queries(output_path, queries)
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
    pgm_epsilon_candidates: list[int],
    k_candidates: list[int],
    delta_candidates: list[int],
    folds: int,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    workload_dir.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    worst_case_rows: list[dict[str, object]] = []
    lookup_latency_rows: list[dict[str, object]] = []
    pgm_sensitivity_rows: list[dict[str, object]] = []

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
        pgm_epsilon, pgm_grid = tune_pgm_grid(
            property_name=property_name,
            records=records,
            train_records=split.train,
            epsilon_candidates=pgm_epsilon_candidates,
        )
        pgm_sensitivity_rows.extend(_pgm_grid_result_to_row(result) for result in pgm_grid)

        btree_metrics, btree_lookup_rows = _evaluate_btree(
            records,
            queries,
            property_name=property_name,
            order=btree_order,
        )
        rmi_metrics, worst_cases, rmi_lookup_rows = _evaluate_rmi(
            records,
            split.train,
            queries,
            property_name=property_name,
            k=tuned.k,
            delta=tuned.delta,
        )
        pgm_metrics, pgm_lookup_rows = _evaluate_pgm(
            records,
            split.train,
            queries,
            property_name=property_name,
            epsilon=pgm_epsilon,
        )
        worst_case_rows.extend(worst_cases)
        lookup_latency_rows.extend(btree_lookup_rows)
        lookup_latency_rows.extend(rmi_lookup_rows)
        lookup_latency_rows.extend(pgm_lookup_rows)

        main_rows.extend(
            _main_metric_rows(
                property_name=property_name,
                btree_metrics=btree_metrics,
                rmi_metrics=rmi_metrics,
                pgm_metrics=pgm_metrics,
            )
        )

    output_paths = {
        "main_comparison": results_dir / "main_comparison.csv",
        "rmi_hyperparameter_sensitivity": results_dir
        / "rmi_hyperparameter_sensitivity.csv",
        "pgm_hyperparameter_sensitivity": results_dir / "pgm_hyperparameter_sensitivity.csv",
        "rmi_worst_case_queries": results_dir / "rmi_worst_case_queries.csv",
        "lookup_latency_detail": results_dir / "lookup_latency_detail.csv",
    }
    _write_dict_rows(output_paths["main_comparison"], main_rows)
    _write_dict_rows(output_paths["rmi_hyperparameter_sensitivity"], sensitivity_rows)
    _write_dict_rows(output_paths["pgm_hyperparameter_sensitivity"], pgm_sensitivity_rows)
    _write_dict_rows(output_paths["rmi_worst_case_queries"], worst_case_rows)
    _write_dict_rows(output_paths["lookup_latency_detail"], lookup_latency_rows)
    return output_paths


def run_range_experiments(
    *,
    data_dir: Path,
    workload_dir: Path,
    results_dir: Path,
    properties: list[str],
    train_fraction: float,
    query_count: int,
    seed: int,
    btree_order: int,
    pgm_epsilon_candidates: list[int],
    k_candidates: list[int],
    delta_candidates: list[int],
    folds: int,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    workload_dir.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, object]] = []
    worst_case_rows: list[dict[str, object]] = []
    lookup_latency_rows: list[dict[str, object]] = []

    for offset, property_name in enumerate(properties):
        records = read_records_csv(data_dir / f"{property_name}.csv")
        split = split_records(records, train_fraction=train_fraction, seed=seed)
        workload_path = workload_dir / f"{property_name}_range_queries.csv"
        if workload_path.exists():
            queries = read_range_queries(workload_path)
        else:
            queries = sample_range_queries(
                property_name,
                split.test,
                records,
                query_count=query_count,
                seed=seed + offset + 101,
            )
            write_range_queries(workload_path, queries)

        tuned, _ = tune_rmi_grid(
            property_name=property_name,
            records=records,
            train_records=split.train,
            k_candidates=k_candidates,
            delta_candidates=delta_candidates,
            folds=folds,
        )
        pgm_epsilon, _ = tune_pgm_grid(
            property_name=property_name,
            records=records,
            train_records=split.train,
            epsilon_candidates=pgm_epsilon_candidates,
        )

        btree_metrics, btree_lookup_rows = _evaluate_btree_ranges(
            records,
            queries,
            property_name=property_name,
            order=btree_order,
        )
        rmi_metrics, rmi_worst_cases, rmi_lookup_rows = _evaluate_rmi_ranges(
            records,
            split.train,
            queries,
            property_name=property_name,
            k=tuned.k,
            delta=tuned.delta,
        )
        pgm_metrics, pgm_worst_cases, pgm_lookup_rows = _evaluate_pgm_ranges(
            records,
            split.train,
            queries,
            property_name=property_name,
            epsilon=pgm_epsilon,
        )

        worst_case_rows.extend(rmi_worst_cases)
        worst_case_rows.extend(pgm_worst_cases)
        lookup_latency_rows.extend(btree_lookup_rows)
        lookup_latency_rows.extend(rmi_lookup_rows)
        lookup_latency_rows.extend(pgm_lookup_rows)
        main_rows.extend(
            _range_main_metric_rows(
                property_name=property_name,
                btree_metrics=btree_metrics,
                rmi_metrics=rmi_metrics,
                pgm_metrics=pgm_metrics,
            )
        )

    output_paths = {
        "range_main_comparison": results_dir / "range_main_comparison.csv",
        "range_worst_case_queries": results_dir / "range_worst_case_queries.csv",
        "range_lookup_latency_detail": results_dir / "range_lookup_latency_detail.csv",
    }
    _write_dict_rows(output_paths["range_main_comparison"], main_rows)
    _write_dict_rows(output_paths["range_worst_case_queries"], worst_case_rows)
    _write_dict_rows(output_paths["range_lookup_latency_detail"], lookup_latency_rows)
    return output_paths


def _evaluate_btree(
    records: list[PropertyRecord],
    queries: list[PointQuery],
    *,
    property_name: str,
    order: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    evaluation_started = perf_counter()
    build_started = perf_counter()
    index = BTreeIndex(records, order=order)
    build_time_ms = (perf_counter() - build_started) * 1000
    comparisons = math.ceil(math.log2(len(records))) if len(records) > 1 else len(records)
    latencies: list[float] = []
    lookup_rows: list[dict[str, object]] = []

    for query in queries:
        started = perf_counter()
        matches = index.exact(query.value)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        lookup_rows.append(
            {
                "property": property_name,
                "index": "B-Tree",
                "value": query.value,
                "node_id": query.node_id,
                "true_position": query.true_position,
                "predicted_position": "",
                "absolute_error": "",
                "elements_examined": comparisons,
                "covered": True,
                "found": any(record.position == query.true_position for record in matches),
                "lookup_latency_ms": _format_float(latency_ms),
                "k": "",
                "delta": "",
            }
        )

    return (
        {
            "mae": math.nan,
            "elements_examined_avg": float(comparisons),
            "build_time_ms": build_time_ms,
            "execution_time_ms": (perf_counter() - evaluation_started) * 1000,
            "latency_avg_ms": mean(latencies) if latencies else 0.0,
            "latency_min_ms": min(latencies) if latencies else 0.0,
            "latency_max_ms": max(latencies) if latencies else 0.0,
        },
        lookup_rows,
    )


def _evaluate_rmi(
    records: list[PropertyRecord],
    train_records: list[PropertyRecord],
    queries: list[PointQuery],
    *,
    property_name: str,
    k: int,
    delta: int,
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
    evaluation_started = perf_counter()
    build_started = perf_counter()
    index = RMIIndex(records, k=k, delta=delta, training_records=train_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    errors: list[int] = []
    elements_examined: list[int] = []
    covered = 0
    worst_cases: list[dict[str, object]] = []
    lookup_rows: list[dict[str, object]] = []

    for query in queries:
        sampled_record = PropertyRecord(
            value=query.value,
            node_id=query.node_id,
            position=query.true_position,
        )
        started = perf_counter()
        predicted_position = index.predict_position(query.value)
        window_start, window_end = index.prediction_window(query.value)
        matches = index.exact(query.value)
        found = any(record.position == query.true_position for record in matches)
        examined = index.elements_examined(query.value)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)

        error = abs(predicted_position - query.true_position)
        is_covered = index.covers_position(sampled_record)
        errors.append(error)
        elements_examined.append(examined)
        covered += int(is_covered)
        lookup_rows.append(
            {
                "property": property_name,
                "index": "RMI",
                "value": query.value,
                "node_id": query.node_id,
                "true_position": query.true_position,
                "predicted_position": predicted_position,
                "absolute_error": error,
                "elements_examined": examined,
                "covered": is_covered,
                "found": found,
                "lookup_latency_ms": _format_float(latency_ms),
                "k": k,
                "delta": delta,
            }
        )
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
                "window_start": window_start,
                "window_end": window_end,
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
        lookup_rows,
    )


def _evaluate_pgm(
    records: list[PropertyRecord],
    train_records: list[PropertyRecord],
    queries: list[PointQuery],
    *,
    property_name: str,
    epsilon: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    evaluation_started = perf_counter()
    build_started = perf_counter()
    index = StaticPGMIndex(records, epsilon=epsilon, training_records=train_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    errors: list[int] = []
    elements_examined: list[int] = []
    covered = 0
    lookup_rows: list[dict[str, object]] = []

    for query in queries:
        sampled_record = PropertyRecord(
            value=query.value,
            node_id=query.node_id,
            position=query.true_position,
        )
        started = perf_counter()
        predicted_position = index.predict_position(query.value)
        matches = index.exact(query.value)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)

        error = abs(predicted_position - query.true_position)
        examined = index.elements_examined(query.value)
        covered_flag = index.covers_position(sampled_record)
        found = any(record.position == query.true_position for record in matches)
        errors.append(error)
        elements_examined.append(examined)
        covered += int(covered_flag)
        lookup_rows.append(
            {
                "property": property_name,
                "index": "PGM-static",
                "value": query.value,
                "node_id": query.node_id,
                "true_position": query.true_position,
                "predicted_position": predicted_position,
                "absolute_error": error,
                "elements_examined": examined,
                "covered": covered_flag,
                "found": found,
                "lookup_latency_ms": _format_float(latency_ms),
                "k": "",
                "delta": "",
            }
        )

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
            "epsilon": float(epsilon),
        },
        lookup_rows,
    )


def _evaluate_btree_ranges(
    records: list[PropertyRecord],
    queries: list[RangeQuery],
    *,
    property_name: str,
    order: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    evaluation_started = perf_counter()
    build_started = perf_counter()
    index = BTreeIndex(records, order=order)
    build_time_ms = (perf_counter() - build_started) * 1000
    comparisons = 2 * (
        math.ceil(math.log2(len(records))) if len(records) > 1 else len(records)
    )
    latencies: list[float] = []
    lookup_rows: list[dict[str, object]] = []

    for query in queries:
        started = perf_counter()
        matches = index.range(query.minimum_value, query.maximum_value)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        actual_start, actual_end = _range_bounds(matches)
        lookup_rows.append(
            {
                "property": property_name,
                "index": "B-Tree",
                "minimum_value": query.minimum_value,
                "maximum_value": query.maximum_value,
                "true_start_position": query.true_start_position,
                "true_end_position": query.true_end_position,
                "predicted_start_position": actual_start,
                "predicted_end_position": actual_end,
                "start_absolute_error": 0,
                "end_absolute_error": 0,
                "true_result_count": query.true_result_count,
                "result_count": len(matches),
                "result_count_error": 0,
                "binary_search_comparisons": comparisons,
                "exact_range": True,
                "lookup_latency_ms": _format_float(latency_ms),
                "k": "",
                "delta": "",
                "epsilon": "",
            }
        )

    return (
        {
            "start_mae": 0.0,
            "end_mae": 0.0,
            "count_error_avg": 0.0,
            "binary_search_comparisons_avg": float(comparisons),
            "build_time_ms": build_time_ms,
            "execution_time_ms": (perf_counter() - evaluation_started) * 1000,
            "latency_avg_ms": mean(latencies) if latencies else 0.0,
            "latency_min_ms": min(latencies) if latencies else 0.0,
            "latency_max_ms": max(latencies) if latencies else 0.0,
            "exact_range_rate": 1.0 if queries else 0.0,
        },
        lookup_rows,
    )


def _evaluate_rmi_ranges(
    records: list[PropertyRecord],
    train_records: list[PropertyRecord],
    queries: list[RangeQuery],
    *,
    property_name: str,
    k: int,
    delta: int,
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
    evaluation_started = perf_counter()
    build_started = perf_counter()
    index = RMIIndex(records, k=k, delta=delta, training_records=train_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    start_errors: list[int] = []
    end_errors: list[int] = []
    count_errors: list[int] = []
    comparisons: list[int] = []
    exact_matches = 0
    worst_cases: list[dict[str, object]] = []
    lookup_rows: list[dict[str, object]] = []

    for query in queries:
        started = perf_counter()
        matches = index.range(query.minimum_value, query.maximum_value)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)

        actual_start, actual_end = _range_bounds(matches)
        start_error = abs(actual_start - query.true_start_position)
        end_error = abs(actual_end - query.true_end_position)
        count_error = abs(len(matches) - query.true_result_count)
        comparison_count = (
            index.boundary_search_comparisons(query.minimum_value)
            + index.boundary_search_comparisons(query.maximum_value)
        )
        exact_range = (
            actual_start == query.true_start_position
            and actual_end == query.true_end_position
            and len(matches) == query.true_result_count
        )
        exact_matches += int(exact_range)
        start_errors.append(start_error)
        end_errors.append(end_error)
        count_errors.append(count_error)
        comparisons.append(comparison_count)
        lookup_rows.append(
            {
                "property": property_name,
                "index": "RMI",
                "minimum_value": query.minimum_value,
                "maximum_value": query.maximum_value,
                "true_start_position": query.true_start_position,
                "true_end_position": query.true_end_position,
                "predicted_start_position": actual_start,
                "predicted_end_position": actual_end,
                "start_absolute_error": start_error,
                "end_absolute_error": end_error,
                "true_result_count": query.true_result_count,
                "result_count": len(matches),
                "result_count_error": count_error,
                "binary_search_comparisons": comparison_count,
                "exact_range": exact_range,
                "lookup_latency_ms": _format_float(latency_ms),
                "k": k,
                "delta": delta,
                "epsilon": "",
            }
        )
        worst_cases.append(
            {
                "property": property_name,
                "index": "RMI",
                "minimum_value": query.minimum_value,
                "maximum_value": query.maximum_value,
                "true_start_position": query.true_start_position,
                "true_end_position": query.true_end_position,
                "predicted_start_position": actual_start,
                "predicted_end_position": actual_end,
                "start_absolute_error": start_error,
                "end_absolute_error": end_error,
                "true_result_count": query.true_result_count,
                "result_count": len(matches),
                "result_count_error": count_error,
                "binary_search_comparisons": comparison_count,
                "exact_range": exact_range,
                "k": k,
                "delta": delta,
                "epsilon": "",
            }
        )

    worst_cases.sort(
        key=lambda row: (
            int(row["start_absolute_error"]) + int(row["end_absolute_error"]),
            int(row["result_count_error"]),
        ),
        reverse=True,
    )
    return (
        {
            "start_mae": mean(start_errors) if start_errors else 0.0,
            "end_mae": mean(end_errors) if end_errors else 0.0,
            "count_error_avg": mean(count_errors) if count_errors else 0.0,
            "binary_search_comparisons_avg": mean(comparisons) if comparisons else 0.0,
            "build_time_ms": build_time_ms,
            "execution_time_ms": (perf_counter() - evaluation_started) * 1000,
            "latency_avg_ms": mean(latencies) if latencies else 0.0,
            "latency_min_ms": min(latencies) if latencies else 0.0,
            "latency_max_ms": max(latencies) if latencies else 0.0,
            "exact_range_rate": exact_matches / len(queries) if queries else 0.0,
            "k": float(k),
            "delta": float(delta),
        },
        worst_cases[:20],
        lookup_rows,
    )


def _evaluate_pgm_ranges(
    records: list[PropertyRecord],
    train_records: list[PropertyRecord],
    queries: list[RangeQuery],
    *,
    property_name: str,
    epsilon: int,
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
    evaluation_started = perf_counter()
    build_started = perf_counter()
    index = StaticPGMIndex(records, epsilon=epsilon, training_records=train_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    start_errors: list[int] = []
    end_errors: list[int] = []
    count_errors: list[int] = []
    comparisons: list[int] = []
    exact_matches = 0
    worst_cases: list[dict[str, object]] = []
    lookup_rows: list[dict[str, object]] = []

    for query in queries:
        started = perf_counter()
        matches = index.range(query.minimum_value, query.maximum_value)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)

        actual_start, actual_end = _range_bounds(matches)
        start_error = abs(actual_start - query.true_start_position)
        end_error = abs(actual_end - query.true_end_position)
        count_error = abs(len(matches) - query.true_result_count)
        comparison_count = (
            index.boundary_search_comparisons(query.minimum_value)
            + index.boundary_search_comparisons(query.maximum_value)
        )
        exact_range = (
            actual_start == query.true_start_position
            and actual_end == query.true_end_position
            and len(matches) == query.true_result_count
        )
        exact_matches += int(exact_range)
        start_errors.append(start_error)
        end_errors.append(end_error)
        count_errors.append(count_error)
        comparisons.append(comparison_count)
        lookup_rows.append(
            {
                "property": property_name,
                "index": "PGM-static",
                "minimum_value": query.minimum_value,
                "maximum_value": query.maximum_value,
                "true_start_position": query.true_start_position,
                "true_end_position": query.true_end_position,
                "predicted_start_position": actual_start,
                "predicted_end_position": actual_end,
                "start_absolute_error": start_error,
                "end_absolute_error": end_error,
                "true_result_count": query.true_result_count,
                "result_count": len(matches),
                "result_count_error": count_error,
                "binary_search_comparisons": comparison_count,
                "exact_range": exact_range,
                "lookup_latency_ms": _format_float(latency_ms),
                "k": "",
                "delta": "",
                "epsilon": epsilon,
            }
        )
        worst_cases.append(
            {
                "property": property_name,
                "index": "PGM-static",
                "minimum_value": query.minimum_value,
                "maximum_value": query.maximum_value,
                "true_start_position": query.true_start_position,
                "true_end_position": query.true_end_position,
                "predicted_start_position": actual_start,
                "predicted_end_position": actual_end,
                "start_absolute_error": start_error,
                "end_absolute_error": end_error,
                "true_result_count": query.true_result_count,
                "result_count": len(matches),
                "result_count_error": count_error,
                "binary_search_comparisons": comparison_count,
                "exact_range": exact_range,
                "k": "",
                "delta": "",
                "epsilon": epsilon,
            }
        )

    worst_cases.sort(
        key=lambda row: (
            int(row["start_absolute_error"]) + int(row["end_absolute_error"]),
            int(row["result_count_error"]),
        ),
        reverse=True,
    )
    return (
        {
            "start_mae": mean(start_errors) if start_errors else 0.0,
            "end_mae": mean(end_errors) if end_errors else 0.0,
            "count_error_avg": mean(count_errors) if count_errors else 0.0,
            "binary_search_comparisons_avg": mean(comparisons) if comparisons else 0.0,
            "build_time_ms": build_time_ms,
            "execution_time_ms": (perf_counter() - evaluation_started) * 1000,
            "latency_avg_ms": mean(latencies) if latencies else 0.0,
            "latency_min_ms": min(latencies) if latencies else 0.0,
            "latency_max_ms": max(latencies) if latencies else 0.0,
            "exact_range_rate": exact_matches / len(queries) if queries else 0.0,
            "epsilon": float(epsilon),
        },
        worst_cases[:20],
        lookup_rows,
    )


def _main_metric_rows(
    *,
    property_name: str,
    btree_metrics: dict[str, float],
    rmi_metrics: dict[str, float],
    pgm_metrics: dict[str, float],
) -> list[dict[str, object]]:
    return [
        {
            "property": property_name,
            "metric": "MAE (positions)",
            "B-Tree": "",
            "RMI": _format_float(rmi_metrics["mae"]),
            "PGM-static": _format_float(pgm_metrics["mae"]),
        },
        {
            "property": property_name,
            "metric": "Binary-search comparisons (avg)",
            "B-Tree": _format_float(btree_metrics["elements_examined_avg"]),
            "RMI": _format_float(rmi_metrics["elements_examined_avg"]),
            "PGM-static": _format_float(pgm_metrics["elements_examined_avg"]),
        },
        {
            "property": property_name,
            "metric": "Index build time (ms)",
            "B-Tree": _format_float(btree_metrics["build_time_ms"]),
            "RMI": _format_float(rmi_metrics["build_time_ms"]),
            "PGM-static": _format_float(pgm_metrics["build_time_ms"]),
        },
        {
            "property": property_name,
            "metric": "Evaluation execution time (ms)",
            "B-Tree": _format_float(btree_metrics["execution_time_ms"]),
            "RMI": _format_float(rmi_metrics["execution_time_ms"]),
            "PGM-static": _format_float(pgm_metrics["execution_time_ms"]),
        },
        {
            "property": property_name,
            "metric": "Lookup latency (ms, avg)",
            "B-Tree": _format_float(btree_metrics["latency_avg_ms"]),
            "RMI": _format_float(rmi_metrics["latency_avg_ms"]),
            "PGM-static": _format_float(pgm_metrics["latency_avg_ms"]),
        },
        {
            "property": property_name,
            "metric": "Lookup latency (ms, min)",
            "B-Tree": _format_float(btree_metrics["latency_min_ms"]),
            "RMI": _format_float(rmi_metrics["latency_min_ms"]),
            "PGM-static": _format_float(pgm_metrics["latency_min_ms"]),
        },
        {
            "property": property_name,
            "metric": "Lookup latency (ms, max)",
            "B-Tree": _format_float(btree_metrics["latency_max_ms"]),
            "RMI": _format_float(rmi_metrics["latency_max_ms"]),
            "PGM-static": _format_float(pgm_metrics["latency_max_ms"]),
        },
        {
            "property": property_name,
            "metric": "RMI coverage",
            "B-Tree": "",
            "RMI": _format_float(rmi_metrics["coverage"]),
            "PGM-static": "",
        },
        {
            "property": property_name,
            "metric": "RMI k",
            "B-Tree": "",
            "RMI": int(rmi_metrics["k"]),
            "PGM-static": "",
        },
        {
            "property": property_name,
            "metric": "RMI delta",
            "B-Tree": "",
            "RMI": int(rmi_metrics["delta"]),
            "PGM-static": "",
        },
        {
            "property": property_name,
            "metric": "PGM coverage",
            "B-Tree": "",
            "RMI": "",
            "PGM-static": _format_float(pgm_metrics["coverage"]),
        },
        {
            "property": property_name,
            "metric": "PGM epsilon",
            "B-Tree": "",
            "RMI": "",
            "PGM-static": int(pgm_metrics["epsilon"]),
        },
    ]


def _range_main_metric_rows(
    *,
    property_name: str,
    btree_metrics: dict[str, float],
    rmi_metrics: dict[str, float],
    pgm_metrics: dict[str, float],
) -> list[dict[str, object]]:
    return [
        {
            "property": property_name,
            "metric": "Start boundary MAE (positions)",
            "B-Tree": _format_float(btree_metrics["start_mae"]),
            "RMI": _format_float(rmi_metrics["start_mae"]),
            "PGM-static": _format_float(pgm_metrics["start_mae"]),
        },
        {
            "property": property_name,
            "metric": "End boundary MAE (positions)",
            "B-Tree": _format_float(btree_metrics["end_mae"]),
            "RMI": _format_float(rmi_metrics["end_mae"]),
            "PGM-static": _format_float(pgm_metrics["end_mae"]),
        },
        {
            "property": property_name,
            "metric": "Result count error (avg)",
            "B-Tree": _format_float(btree_metrics["count_error_avg"]),
            "RMI": _format_float(rmi_metrics["count_error_avg"]),
            "PGM-static": _format_float(pgm_metrics["count_error_avg"]),
        },
        {
            "property": property_name,
            "metric": "Binary-search comparisons (avg)",
            "B-Tree": _format_float(btree_metrics["binary_search_comparisons_avg"]),
            "RMI": _format_float(rmi_metrics["binary_search_comparisons_avg"]),
            "PGM-static": _format_float(pgm_metrics["binary_search_comparisons_avg"]),
        },
        {
            "property": property_name,
            "metric": "Index build time (ms)",
            "B-Tree": _format_float(btree_metrics["build_time_ms"]),
            "RMI": _format_float(rmi_metrics["build_time_ms"]),
            "PGM-static": _format_float(pgm_metrics["build_time_ms"]),
        },
        {
            "property": property_name,
            "metric": "Evaluation execution time (ms)",
            "B-Tree": _format_float(btree_metrics["execution_time_ms"]),
            "RMI": _format_float(rmi_metrics["execution_time_ms"]),
            "PGM-static": _format_float(pgm_metrics["execution_time_ms"]),
        },
        {
            "property": property_name,
            "metric": "Lookup latency (ms, avg)",
            "B-Tree": _format_float(btree_metrics["latency_avg_ms"]),
            "RMI": _format_float(rmi_metrics["latency_avg_ms"]),
            "PGM-static": _format_float(pgm_metrics["latency_avg_ms"]),
        },
        {
            "property": property_name,
            "metric": "Lookup latency (ms, min)",
            "B-Tree": _format_float(btree_metrics["latency_min_ms"]),
            "RMI": _format_float(rmi_metrics["latency_min_ms"]),
            "PGM-static": _format_float(pgm_metrics["latency_min_ms"]),
        },
        {
            "property": property_name,
            "metric": "Lookup latency (ms, max)",
            "B-Tree": _format_float(btree_metrics["latency_max_ms"]),
            "RMI": _format_float(rmi_metrics["latency_max_ms"]),
            "PGM-static": _format_float(pgm_metrics["latency_max_ms"]),
        },
        {
            "property": property_name,
            "metric": "Exact range correctness",
            "B-Tree": _format_float(btree_metrics["exact_range_rate"]),
            "RMI": _format_float(rmi_metrics["exact_range_rate"]),
            "PGM-static": _format_float(pgm_metrics["exact_range_rate"]),
        },
        {
            "property": property_name,
            "metric": "RMI k",
            "B-Tree": "",
            "RMI": int(rmi_metrics["k"]),
            "PGM-static": "",
        },
        {
            "property": property_name,
            "metric": "RMI delta",
            "B-Tree": "",
            "RMI": int(rmi_metrics["delta"]),
            "PGM-static": "",
        },
        {
            "property": property_name,
            "metric": "PGM epsilon",
            "B-Tree": "",
            "RMI": "",
            "PGM-static": int(pgm_metrics["epsilon"]),
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


def _pgm_grid_result_to_row(result: PGMGridResult) -> dict[str, object]:
    return {
        "property": result.property_name,
        "epsilon": result.epsilon,
        "validation_mae": _format_float(result.validation_mae),
        "validation_elements_examined_avg": _format_float(
            result.validation_elements_examined_avg
        ),
        "validation_coverage": _format_float(result.validation_coverage),
        "validation_miss_count": result.validation_miss_count,
    }


def tune_pgm_grid(
    *,
    property_name: str,
    records: list[PropertyRecord],
    train_records: list[PropertyRecord],
    epsilon_candidates: list[int],
) -> tuple[int, list[PGMGridResult]]:
    results: list[PGMGridResult] = []
    for epsilon in epsilon_candidates:
        if epsilon < 1:
            continue
        index = StaticPGMIndex(records, epsilon=epsilon, training_records=train_records)
        errors: list[int] = []
        elements_examined: list[int] = []
        covered = 0
        for record in train_records:
            errors.append(index.prediction_error(record))
            elements_examined.append(index.elements_examined(record.value))
            covered += int(index.covers_position(record))
        total = len(train_records)
        results.append(
            PGMGridResult(
                property_name=property_name,
                epsilon=epsilon,
                validation_mae=mean(errors) if errors else 0.0,
                validation_elements_examined_avg=mean(elements_examined)
                if elements_examined
                else 0.0,
                validation_coverage=covered / total if total else 0.0,
                validation_miss_count=total - covered,
            )
        )

    if not results:
        raise ValueError("No valid PGM epsilon configurations were evaluated.")

    best = min(
        results,
        key=lambda result: (
            -result.validation_coverage,
            result.validation_elements_examined_avg,
            result.validation_mae,
            result.epsilon,
        ),
    )
    return best.epsilon, results


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _range_bounds(matches: list[PropertyRecord]) -> tuple[int, int]:
    if not matches:
        return -1, -1
    return matches[0].position, matches[-1].position


def _parse_numeric(value: str) -> NumericValue:
    return float(value) if "." in value else int(value)


def _format_float(value: float) -> str:
    return f"{value:.6f}"
