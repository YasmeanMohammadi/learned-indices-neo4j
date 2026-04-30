import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from learned_indices_neo4j.btree_index import BTreeIndex
from learned_indices_neo4j.experiments import (
    RangeQuery,
    PointQuery,
    sample_point_queries,
    sample_range_queries,
    split_records,
    tune_pgm_grid,
    tune_rmi_grid,
)
from learned_indices_neo4j.io import read_records_csv
from learned_indices_neo4j.pgm_index import StaticPGMIndex
from learned_indices_neo4j.records import PropertyRecord, preprocess_pairs
from learned_indices_neo4j.rmi_index import RMIIndex


@dataclass(frozen=True)
class ShiftScenario:
    property_name: str
    scenario_name: str
    shift_fraction: float
    shifted_records: list[PropertyRecord]
    inserted_records: list[PropertyRecord]


def generate_shift_scenarios(
    property_name: str,
    base_records: list[PropertyRecord],
    *,
    shift_fractions: list[float],
    seed: int,
) -> list[ShiftScenario]:
    scenarios: list[ShiftScenario] = []
    for offset, shift_fraction in enumerate(shift_fractions):
        if shift_fraction <= 0:
            continue
        count = max(1, int(len(base_records) * shift_fraction))
        randomizer = random.Random(seed + offset + 1)

        for scenario_name, pairs in _scenario_pairs(
            property_name,
            base_records,
            count=count,
            randomizer=randomizer,
        ):
            shifted_records = _merge_pairs(base_records, pairs)
            inserted_ids = {node_id for _, node_id in pairs}
            inserted_records = [
                record for record in shifted_records if record.node_id in inserted_ids
            ]
            scenarios.append(
                ShiftScenario(
                    property_name=property_name,
                    scenario_name=scenario_name,
                    shift_fraction=shift_fraction,
                    shifted_records=shifted_records,
                    inserted_records=inserted_records,
                )
            )

    return scenarios


def run_distribution_shift_experiment(
    *,
    data_dir: Path,
    results_dir: Path,
    properties: list[str],
    train_fraction: float,
    point_query_count: int,
    range_query_count: int,
    shift_fractions: list[float],
    seed: int,
    btree_order: int,
    pgm_epsilon_candidates: list[int],
    k_candidates: list[int],
    delta_candidates: list[int],
    folds: int,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    range_rows: list[dict[str, object]] = []

    for offset, property_name in enumerate(properties):
        base_records = read_records_csv(data_dir / f"{property_name}.csv")
        base_split = split_records(
            base_records,
            train_fraction=train_fraction,
            seed=seed + offset,
        )
        base_point_queries = sample_point_queries(
            property_name,
            base_split.test,
            query_count=point_query_count,
            seed=seed + offset + 100,
        )
        base_range_queries = sample_range_queries(
            property_name,
            base_split.test,
            base_records,
            query_count=range_query_count,
            seed=seed + offset + 200,
        )
        base_rmi, _ = tune_rmi_grid(
            property_name=property_name,
            records=base_records,
            train_records=base_split.train,
            k_candidates=k_candidates,
            delta_candidates=delta_candidates,
            folds=folds,
        )
        base_pgm, _ = tune_pgm_grid(
            property_name=property_name,
            records=base_records,
            train_records=base_split.train,
            epsilon_candidates=pgm_epsilon_candidates,
        )

        point_rows.extend(
            _phase_point_rows(
                property_name=property_name,
                scenario_name="baseline",
                shift_fraction=0.0,
                phase="baseline",
                records=base_records,
                queries=base_point_queries,
                btree_order=btree_order,
                rmi_train_records=base_split.train,
                rmi_k=base_rmi.k,
                rmi_delta=base_rmi.delta,
                pgm_train_records=base_split.train,
                pgm_epsilon=base_pgm,
            )
        )
        range_rows.extend(
            _phase_range_rows(
                property_name=property_name,
                scenario_name="baseline",
                shift_fraction=0.0,
                phase="baseline",
                records=base_records,
                queries=base_range_queries,
                btree_order=btree_order,
                rmi_train_records=base_split.train,
                rmi_k=base_rmi.k,
                rmi_delta=base_rmi.delta,
                pgm_train_records=base_split.train,
                pgm_epsilon=base_pgm,
            )
        )

        for scenario in generate_shift_scenarios(
            property_name,
            base_records,
            shift_fractions=shift_fractions,
            seed=seed + offset + 300,
        ):
            shifted_split = split_records(
                scenario.shifted_records,
                train_fraction=train_fraction,
                seed=seed + offset + 400,
            )
            point_queries = sample_point_queries(
                property_name,
                scenario.inserted_records or shifted_split.test,
                query_count=point_query_count,
                seed=seed + offset + 500,
            )
            range_queries = sample_range_queries(
                property_name,
                scenario.inserted_records or shifted_split.test,
                scenario.shifted_records,
                query_count=range_query_count,
                seed=seed + offset + 600,
            )

            point_rows.extend(
                _phase_point_rows(
                    property_name=property_name,
                    scenario_name=scenario.scenario_name,
                    shift_fraction=scenario.shift_fraction,
                    phase="shifted_stale_models",
                    records=scenario.shifted_records,
                    queries=point_queries,
                    btree_order=btree_order,
                    rmi_train_records=base_split.train,
                    rmi_k=base_rmi.k,
                    rmi_delta=base_rmi.delta,
                    pgm_train_records=base_split.train,
                    pgm_epsilon=base_pgm,
                )
            )
            range_rows.extend(
                _phase_range_rows(
                    property_name=property_name,
                    scenario_name=scenario.scenario_name,
                    shift_fraction=scenario.shift_fraction,
                    phase="shifted_stale_models",
                    records=scenario.shifted_records,
                    queries=range_queries,
                    btree_order=btree_order,
                    rmi_train_records=base_split.train,
                    rmi_k=base_rmi.k,
                    rmi_delta=base_rmi.delta,
                    pgm_train_records=base_split.train,
                    pgm_epsilon=base_pgm,
                )
            )

            shifted_rmi, _ = tune_rmi_grid(
                property_name=property_name,
                records=scenario.shifted_records,
                train_records=shifted_split.train,
                k_candidates=k_candidates,
                delta_candidates=delta_candidates,
                folds=folds,
            )
            shifted_pgm, _ = tune_pgm_grid(
                property_name=property_name,
                records=scenario.shifted_records,
                train_records=shifted_split.train,
                epsilon_candidates=pgm_epsilon_candidates,
            )
            point_rows.extend(
                _phase_point_rows(
                    property_name=property_name,
                    scenario_name=scenario.scenario_name,
                    shift_fraction=scenario.shift_fraction,
                    phase="shifted_retrained_models",
                    records=scenario.shifted_records,
                    queries=point_queries,
                    btree_order=btree_order,
                    rmi_train_records=shifted_split.train,
                    rmi_k=shifted_rmi.k,
                    rmi_delta=shifted_rmi.delta,
                    pgm_train_records=shifted_split.train,
                    pgm_epsilon=shifted_pgm,
                )
            )
            range_rows.extend(
                _phase_range_rows(
                    property_name=property_name,
                    scenario_name=scenario.scenario_name,
                    shift_fraction=scenario.shift_fraction,
                    phase="shifted_retrained_models",
                    records=scenario.shifted_records,
                    queries=range_queries,
                    btree_order=btree_order,
                    rmi_train_records=shifted_split.train,
                    rmi_k=shifted_rmi.k,
                    rmi_delta=shifted_rmi.delta,
                    pgm_train_records=shifted_split.train,
                    pgm_epsilon=shifted_pgm,
                )
            )

    summary_rows.extend(point_rows)
    summary_rows.extend(range_rows)
    output_paths = {
        "distribution_shift_summary": results_dir / "distribution_shift_summary.csv",
        "distribution_shift_point": results_dir / "distribution_shift_point.csv",
        "distribution_shift_range": results_dir / "distribution_shift_range.csv",
    }
    _write_rows(output_paths["distribution_shift_summary"], summary_rows)
    _write_rows(output_paths["distribution_shift_point"], point_rows)
    _write_rows(output_paths["distribution_shift_range"], range_rows)
    return output_paths


def _phase_point_rows(
    *,
    property_name: str,
    scenario_name: str,
    shift_fraction: float,
    phase: str,
    records: list[PropertyRecord],
    queries: list[PointQuery],
    btree_order: int,
    rmi_train_records: list[PropertyRecord],
    rmi_k: int,
    rmi_delta: int,
    pgm_train_records: list[PropertyRecord],
    pgm_epsilon: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(
        _point_summary_row(
            property_name=property_name,
            scenario_name=scenario_name,
            shift_fraction=shift_fraction,
            phase=phase,
            index_name="B-Tree",
            metrics=_evaluate_point_btree(records, queries, order=btree_order),
        )
    )
    rows.append(
        _point_summary_row(
            property_name=property_name,
            scenario_name=scenario_name,
            shift_fraction=shift_fraction,
            phase=phase,
            index_name="RMI",
            metrics=_evaluate_point_rmi(
                records,
                queries,
                training_records=rmi_train_records,
                k=rmi_k,
                delta=rmi_delta,
            ),
        )
    )
    rows.append(
        _point_summary_row(
            property_name=property_name,
            scenario_name=scenario_name,
            shift_fraction=shift_fraction,
            phase=phase,
            index_name="PGM-static",
            metrics=_evaluate_point_pgm(
                records,
                queries,
                training_records=pgm_train_records,
                epsilon=pgm_epsilon,
            ),
        )
    )
    return rows


def _phase_range_rows(
    *,
    property_name: str,
    scenario_name: str,
    shift_fraction: float,
    phase: str,
    records: list[PropertyRecord],
    queries: list[RangeQuery],
    btree_order: int,
    rmi_train_records: list[PropertyRecord],
    rmi_k: int,
    rmi_delta: int,
    pgm_train_records: list[PropertyRecord],
    pgm_epsilon: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(
        _range_summary_row(
            property_name=property_name,
            scenario_name=scenario_name,
            shift_fraction=shift_fraction,
            phase=phase,
            index_name="B-Tree",
            metrics=_evaluate_range_btree(records, queries, order=btree_order),
        )
    )
    rows.append(
        _range_summary_row(
            property_name=property_name,
            scenario_name=scenario_name,
            shift_fraction=shift_fraction,
            phase=phase,
            index_name="RMI",
            metrics=_evaluate_range_rmi(
                records,
                queries,
                training_records=rmi_train_records,
                k=rmi_k,
                delta=rmi_delta,
            ),
        )
    )
    rows.append(
        _range_summary_row(
            property_name=property_name,
            scenario_name=scenario_name,
            shift_fraction=shift_fraction,
            phase=phase,
            index_name="PGM-static",
            metrics=_evaluate_range_pgm(
                records,
                queries,
                training_records=pgm_train_records,
                epsilon=pgm_epsilon,
            ),
        )
    )
    return rows


def _evaluate_point_btree(
    records: list[PropertyRecord],
    queries: list[PointQuery],
    *,
    order: int,
) -> dict[str, float]:
    started = perf_counter()
    build_started = perf_counter()
    index = BTreeIndex(records, order=order)
    build_time_ms = (perf_counter() - build_started) * 1000
    comparisons = math.ceil(math.log2(len(records))) if len(records) > 1 else len(records)
    latencies: list[float] = []
    found = 0
    for query in queries:
        lookup_started = perf_counter()
        matches = index.exact(query.value)
        latencies.append((perf_counter() - lookup_started) * 1000)
        found += int(any(record.position == query.true_position for record in matches))
    return {
        "mae": math.nan,
        "coverage": 1.0 if queries else 0.0,
        "binary_search_comparisons_avg": float(comparisons),
        "lookup_latency_avg_ms": mean(latencies) if latencies else 0.0,
        "lookup_latency_min_ms": min(latencies) if latencies else 0.0,
        "lookup_latency_max_ms": max(latencies) if latencies else 0.0,
        "build_time_ms": build_time_ms,
        "execution_time_ms": (perf_counter() - started) * 1000,
        "exact_correctness": found / len(queries) if queries else 0.0,
        "k": math.nan,
        "delta": math.nan,
        "epsilon": math.nan,
    }


def _evaluate_point_rmi(
    records: list[PropertyRecord],
    queries: list[PointQuery],
    *,
    training_records: list[PropertyRecord],
    k: int,
    delta: int,
) -> dict[str, float]:
    started = perf_counter()
    build_started = perf_counter()
    index = RMIIndex(records, k=k, delta=delta, training_records=training_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    errors: list[int] = []
    comparisons: list[int] = []
    covered = 0
    found = 0
    for query in queries:
        record = PropertyRecord(value=query.value, node_id=query.node_id, position=query.true_position)
        lookup_started = perf_counter()
        matches = index.exact(query.value)
        latencies.append((perf_counter() - lookup_started) * 1000)
        errors.append(abs(index.predict_position(query.value) - query.true_position))
        comparisons.append(index.elements_examined(query.value))
        covered += int(index.covers_position(record))
        found += int(any(match.position == query.true_position for match in matches))
    return {
        "mae": mean(errors) if errors else 0.0,
        "coverage": covered / len(queries) if queries else 0.0,
        "binary_search_comparisons_avg": mean(comparisons) if comparisons else 0.0,
        "lookup_latency_avg_ms": mean(latencies) if latencies else 0.0,
        "lookup_latency_min_ms": min(latencies) if latencies else 0.0,
        "lookup_latency_max_ms": max(latencies) if latencies else 0.0,
        "build_time_ms": build_time_ms,
        "execution_time_ms": (perf_counter() - started) * 1000,
        "exact_correctness": found / len(queries) if queries else 0.0,
        "k": float(k),
        "delta": float(delta),
        "epsilon": math.nan,
    }


def _evaluate_point_pgm(
    records: list[PropertyRecord],
    queries: list[PointQuery],
    *,
    training_records: list[PropertyRecord],
    epsilon: int,
) -> dict[str, float]:
    started = perf_counter()
    build_started = perf_counter()
    index = StaticPGMIndex(records, epsilon=epsilon, training_records=training_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    errors: list[int] = []
    comparisons: list[int] = []
    covered = 0
    found = 0
    for query in queries:
        record = PropertyRecord(value=query.value, node_id=query.node_id, position=query.true_position)
        lookup_started = perf_counter()
        matches = index.exact(query.value)
        latencies.append((perf_counter() - lookup_started) * 1000)
        errors.append(abs(index.predict_position(query.value) - query.true_position))
        comparisons.append(index.elements_examined(query.value))
        covered += int(index.covers_position(record))
        found += int(any(match.position == query.true_position for match in matches))
    return {
        "mae": mean(errors) if errors else 0.0,
        "coverage": covered / len(queries) if queries else 0.0,
        "binary_search_comparisons_avg": mean(comparisons) if comparisons else 0.0,
        "lookup_latency_avg_ms": mean(latencies) if latencies else 0.0,
        "lookup_latency_min_ms": min(latencies) if latencies else 0.0,
        "lookup_latency_max_ms": max(latencies) if latencies else 0.0,
        "build_time_ms": build_time_ms,
        "execution_time_ms": (perf_counter() - started) * 1000,
        "exact_correctness": found / len(queries) if queries else 0.0,
        "k": math.nan,
        "delta": math.nan,
        "epsilon": float(epsilon),
    }


def _evaluate_range_btree(
    records: list[PropertyRecord],
    queries: list[RangeQuery],
    *,
    order: int,
) -> dict[str, float]:
    started = perf_counter()
    build_started = perf_counter()
    index = BTreeIndex(records, order=order)
    build_time_ms = (perf_counter() - build_started) * 1000
    comparisons = 2 * (math.ceil(math.log2(len(records))) if len(records) > 1 else len(records))
    latencies: list[float] = []
    for query in queries:
        lookup_started = perf_counter()
        index.range(query.minimum_value, query.maximum_value)
        latencies.append((perf_counter() - lookup_started) * 1000)
    return {
        "start_mae": 0.0,
        "end_mae": 0.0,
        "result_count_error_avg": 0.0,
        "binary_search_comparisons_avg": float(comparisons),
        "lookup_latency_avg_ms": mean(latencies) if latencies else 0.0,
        "lookup_latency_min_ms": min(latencies) if latencies else 0.0,
        "lookup_latency_max_ms": max(latencies) if latencies else 0.0,
        "build_time_ms": build_time_ms,
        "execution_time_ms": (perf_counter() - started) * 1000,
        "exact_correctness": 1.0 if queries else 0.0,
        "k": math.nan,
        "delta": math.nan,
        "epsilon": math.nan,
    }


def _evaluate_range_rmi(
    records: list[PropertyRecord],
    queries: list[RangeQuery],
    *,
    training_records: list[PropertyRecord],
    k: int,
    delta: int,
) -> dict[str, float]:
    started = perf_counter()
    build_started = perf_counter()
    index = RMIIndex(records, k=k, delta=delta, training_records=training_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    start_errors: list[int] = []
    end_errors: list[int] = []
    count_errors: list[int] = []
    comparisons: list[int] = []
    exact = 0
    for query in queries:
        lookup_started = perf_counter()
        matches = index.range(query.minimum_value, query.maximum_value)
        latencies.append((perf_counter() - lookup_started) * 1000)
        actual_start, actual_end = _range_bounds(matches)
        start_errors.append(abs(actual_start - query.true_start_position))
        end_errors.append(abs(actual_end - query.true_end_position))
        count_errors.append(abs(len(matches) - query.true_result_count))
        comparisons.append(
            index.boundary_search_comparisons(query.minimum_value)
            + index.boundary_search_comparisons(query.maximum_value)
        )
        exact += int(
            actual_start == query.true_start_position
            and actual_end == query.true_end_position
            and len(matches) == query.true_result_count
        )
    return {
        "start_mae": mean(start_errors) if start_errors else 0.0,
        "end_mae": mean(end_errors) if end_errors else 0.0,
        "result_count_error_avg": mean(count_errors) if count_errors else 0.0,
        "binary_search_comparisons_avg": mean(comparisons) if comparisons else 0.0,
        "lookup_latency_avg_ms": mean(latencies) if latencies else 0.0,
        "lookup_latency_min_ms": min(latencies) if latencies else 0.0,
        "lookup_latency_max_ms": max(latencies) if latencies else 0.0,
        "build_time_ms": build_time_ms,
        "execution_time_ms": (perf_counter() - started) * 1000,
        "exact_correctness": exact / len(queries) if queries else 0.0,
        "k": float(k),
        "delta": float(delta),
        "epsilon": math.nan,
    }


def _evaluate_range_pgm(
    records: list[PropertyRecord],
    queries: list[RangeQuery],
    *,
    training_records: list[PropertyRecord],
    epsilon: int,
) -> dict[str, float]:
    started = perf_counter()
    build_started = perf_counter()
    index = StaticPGMIndex(records, epsilon=epsilon, training_records=training_records)
    build_time_ms = (perf_counter() - build_started) * 1000
    latencies: list[float] = []
    start_errors: list[int] = []
    end_errors: list[int] = []
    count_errors: list[int] = []
    comparisons: list[int] = []
    exact = 0
    for query in queries:
        lookup_started = perf_counter()
        matches = index.range(query.minimum_value, query.maximum_value)
        latencies.append((perf_counter() - lookup_started) * 1000)
        actual_start, actual_end = _range_bounds(matches)
        start_errors.append(abs(actual_start - query.true_start_position))
        end_errors.append(abs(actual_end - query.true_end_position))
        count_errors.append(abs(len(matches) - query.true_result_count))
        comparisons.append(
            index.boundary_search_comparisons(query.minimum_value)
            + index.boundary_search_comparisons(query.maximum_value)
        )
        exact += int(
            actual_start == query.true_start_position
            and actual_end == query.true_end_position
            and len(matches) == query.true_result_count
        )
    return {
        "start_mae": mean(start_errors) if start_errors else 0.0,
        "end_mae": mean(end_errors) if end_errors else 0.0,
        "result_count_error_avg": mean(count_errors) if count_errors else 0.0,
        "binary_search_comparisons_avg": mean(comparisons) if comparisons else 0.0,
        "lookup_latency_avg_ms": mean(latencies) if latencies else 0.0,
        "lookup_latency_min_ms": min(latencies) if latencies else 0.0,
        "lookup_latency_max_ms": max(latencies) if latencies else 0.0,
        "build_time_ms": build_time_ms,
        "execution_time_ms": (perf_counter() - started) * 1000,
        "exact_correctness": exact / len(queries) if queries else 0.0,
        "k": math.nan,
        "delta": math.nan,
        "epsilon": float(epsilon),
    }


def _point_summary_row(
    *,
    property_name: str,
    scenario_name: str,
    shift_fraction: float,
    phase: str,
    index_name: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "property": property_name,
        "scenario": scenario_name,
        "shift_fraction": _format_float(shift_fraction),
        "phase": phase,
        "workload": "point",
        "index": index_name,
        "mae": _optional_float(metrics["mae"]),
        "coverage": _optional_float(metrics["coverage"]),
        "start_boundary_mae": "",
        "end_boundary_mae": "",
        "result_count_error_avg": "",
        "binary_search_comparisons_avg": _format_float(
            metrics["binary_search_comparisons_avg"]
        ),
        "lookup_latency_avg_ms": _format_float(metrics["lookup_latency_avg_ms"]),
        "lookup_latency_min_ms": _format_float(metrics["lookup_latency_min_ms"]),
        "lookup_latency_max_ms": _format_float(metrics["lookup_latency_max_ms"]),
        "build_time_ms": _format_float(metrics["build_time_ms"]),
        "execution_time_ms": _format_float(metrics["execution_time_ms"]),
        "exact_correctness": _format_float(metrics["exact_correctness"]),
        "k": _optional_int(metrics["k"]),
        "delta": _optional_int(metrics["delta"]),
        "epsilon": _optional_int(metrics["epsilon"]),
    }


def _range_summary_row(
    *,
    property_name: str,
    scenario_name: str,
    shift_fraction: float,
    phase: str,
    index_name: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "property": property_name,
        "scenario": scenario_name,
        "shift_fraction": _format_float(shift_fraction),
        "phase": phase,
        "workload": "range",
        "index": index_name,
        "mae": "",
        "coverage": "",
        "start_boundary_mae": _format_float(metrics["start_mae"]),
        "end_boundary_mae": _format_float(metrics["end_mae"]),
        "result_count_error_avg": _format_float(metrics["result_count_error_avg"]),
        "binary_search_comparisons_avg": _format_float(
            metrics["binary_search_comparisons_avg"]
        ),
        "lookup_latency_avg_ms": _format_float(metrics["lookup_latency_avg_ms"]),
        "lookup_latency_min_ms": _format_float(metrics["lookup_latency_min_ms"]),
        "lookup_latency_max_ms": _format_float(metrics["lookup_latency_max_ms"]),
        "build_time_ms": _format_float(metrics["build_time_ms"]),
        "execution_time_ms": _format_float(metrics["execution_time_ms"]),
        "exact_correctness": _format_float(metrics["exact_correctness"]),
        "k": _optional_int(metrics["k"]),
        "delta": _optional_int(metrics["delta"]),
        "epsilon": _optional_int(metrics["epsilon"]),
    }


def _scenario_pairs(
    property_name: str,
    base_records: list[PropertyRecord],
    *,
    count: int,
    randomizer: random.Random,
) -> list[tuple[str, list[tuple[int, str]]]]:
    values = [int(record.value) for record in base_records]
    max_value = max(values)
    p75_value = values[int(0.75 * (len(values) - 1))]
    p90_value = values[int(0.9 * (len(values) - 1))]

    if property_name == "year":
        recent_tail = [
            (max_value + 1 + (index % 8), f"shift-year-recent-{index}")
            for index in range(count)
        ]
        duplicate_burst = [
            (p75_value, f"shift-year-dup-{index}") for index in range(count)
        ]
        return [
            ("recent_tail", recent_tail),
            ("duplicate_burst", duplicate_burst),
        ]

    high_tail = [
        (
            int(round(max_value * (1.1 + randomizer.random() * 1.4))),
            f"shift-{property_name}-tail-{index}",
        )
        for index in range(count)
    ]
    duplicate_burst = [
        (p90_value, f"shift-{property_name}-dup-{index}") for index in range(count)
    ]
    return [
        ("high_tail", high_tail),
        ("duplicate_burst", duplicate_burst),
    ]


def _merge_pairs(
    base_records: list[PropertyRecord],
    inserted_pairs: list[tuple[int, str]],
) -> list[PropertyRecord]:
    base_pairs = [(record.value, record.node_id) for record in base_records]
    return preprocess_pairs(base_pairs + inserted_pairs)


def _range_bounds(matches: list[PropertyRecord]) -> tuple[int, int]:
    if not matches:
        return -1, -1
    return matches[0].position, matches[-1].position


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return _format_float(value)


def _optional_int(value: float) -> str:
    if math.isnan(value):
        return ""
    return str(int(value))


def _format_float(value: float) -> str:
    return f"{value:.6f}"
