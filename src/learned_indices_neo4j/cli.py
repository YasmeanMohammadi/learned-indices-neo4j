import argparse
import sys
from pathlib import Path

from learned_indices_neo4j.btree_index import BTreeIndex
from learned_indices_neo4j.config import ExperimentSettings, Neo4jSettings
from learned_indices_neo4j.distribution_shift import run_distribution_shift_experiment
from learned_indices_neo4j.experiments import (
    generate_range_workloads,
    generate_workloads,
    run_experiments,
    run_range_experiments,
)
from learned_indices_neo4j.io import read_records_csv, write_records_csv
from learned_indices_neo4j.neo4j_extractor import (
    EmptyExtractionError,
    PROPERTY_QUERIES,
    create_neo4j_range_indexes,
    extract_property_pairs,
)
from learned_indices_neo4j.pgm_index import StaticPGMIndex
from learned_indices_neo4j.records import preprocess_pairs
from learned_indices_neo4j.rmi_index import RMIIndex, tune_rmi
from learned_indices_neo4j.sorted_array_index import SortedArrayIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learned-indices-neo4j",
        description="Build baseline sorted-array indexes from Neo4j Recommendations properties.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiment.toml"),
        help="Experiment config file with index and tuning hyperparameters.",
    )
    subparsers = parser.add_subparsers(dest="command")

    extract = subparsers.add_parser(
        "extract",
        help="Extract and preprocess Movie.year and Movie.imdbVotes values from Neo4j.",
    )
    extract.add_argument(
        "--property",
        dest="properties",
        choices=sorted(PROPERTY_QUERIES),
        action="append",
        help="Property to extract. May be passed more than once. Defaults to both.",
    )
    extract.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where preprocessed CSV files should be written.",
    )
    extract.add_argument(
        "--database",
        default=None,
        help="Optional Neo4j database name. Defaults to NEO4J_DATABASE or the server default.",
    )

    create_indexes = subparsers.add_parser(
        "create-indexes",
        help="Create Neo4j range indexes for Movie.year and Movie.imdbVotes.",
    )
    create_indexes.add_argument(
        "--property",
        dest="properties",
        choices=sorted(PROPERTY_QUERIES),
        action="append",
        help="Property to index. May be passed more than once. Defaults to both.",
    )
    create_indexes.add_argument(
        "--database",
        default=None,
        help="Optional Neo4j database name. Defaults to NEO4J_DATABASE or the server default.",
    )

    query = subparsers.add_parser(
        "query",
        help="Run exact or range lookups against a preprocessed CSV file.",
    )
    query.add_argument("csv_path", type=Path)
    query.add_argument("--exact", type=float, default=None)
    query.add_argument("--min", dest="minimum", type=float, default=None)
    query.add_argument("--max", dest="maximum", type=float, default=None)
    query.add_argument("--limit", type=int, default=None)
    query.add_argument(
        "--index",
        choices=["sorted-array", "btree", "rmi", "pgm"],
        default="sorted-array",
        help="Local index implementation to use for CSV lookups.",
    )
    query.add_argument(
        "--order",
        type=int,
        default=None,
        help="B-tree order when --index btree is selected.",
    )
    query.add_argument(
        "--pgm-epsilon",
        type=int,
        default=None,
        help="PGM approximation error bound when --index pgm is selected.",
    )
    query.add_argument("--rmi-k", type=int, default=None, help="Number of stage-2 RMI models.")
    query.add_argument(
        "--rmi-delta",
        type=int,
        default=None,
        help="RMI local scan radius. Defaults to calibrated max training error.",
    )
    query.add_argument(
        "--tune-rmi",
        action="store_true",
        help="Tune RMI k and delta with cross-validation before running the query.",
    )
    query.add_argument(
        "--rmi-k-candidates",
        default=None,
        help="Comma-separated k candidates used by --tune-rmi.",
    )
    query.add_argument(
        "--rmi-delta-candidates",
        default=None,
        help="Optional comma-separated delta candidates used by --tune-rmi.",
    )
    query.add_argument("--rmi-folds", type=int, default=None, help="Cross-validation folds.")

    tune = subparsers.add_parser(
        "tune-rmi",
        help="Cross-validate RMI k and delta for a preprocessed CSV file.",
    )
    tune.add_argument("csv_path", type=Path)
    tune.add_argument("--k-candidates", default=None)
    tune.add_argument("--delta-candidates", default=None)
    tune.add_argument("--folds", type=int, default=None)

    workload = subparsers.add_parser(
        "generate-workload",
        help="Generate random point lookup query files from the held-out test split.",
    )
    workload.add_argument("--data-dir", type=Path, default=None)
    workload.add_argument("--workload-dir", type=Path, default=None)
    workload.add_argument("--query-count", type=int, default=None)
    workload.add_argument("--seed", type=int, default=None)
    workload.add_argument("--train-fraction", type=float, default=None)

    range_workload = subparsers.add_parser(
        "generate-range-workload",
        help="Generate random range query files from the held-out test split.",
    )
    range_workload.add_argument("--data-dir", type=Path, default=None)
    range_workload.add_argument("--workload-dir", type=Path, default=None)
    range_workload.add_argument("--query-count", type=int, default=None)
    range_workload.add_argument("--seed", type=int, default=None)
    range_workload.add_argument("--train-fraction", type=float, default=None)

    experiment = subparsers.add_parser(
        "run-experiment",
        help="Tune RMI and write report-ready evaluation CSV files.",
    )
    experiment.add_argument("--data-dir", type=Path, default=None)
    experiment.add_argument("--workload-dir", type=Path, default=None)
    experiment.add_argument("--results-dir", type=Path, default=None)
    experiment.add_argument("--query-count", type=int, default=None)
    experiment.add_argument("--seed", type=int, default=None)
    experiment.add_argument("--train-fraction", type=float, default=None)
    experiment.add_argument("--k-candidates", default=None)
    experiment.add_argument("--delta-candidates", default=None)
    experiment.add_argument("--folds", type=int, default=None)

    range_experiment = subparsers.add_parser(
        "run-range-experiment",
        help="Write report-ready evaluation CSV files for range queries.",
    )
    range_experiment.add_argument("--data-dir", type=Path, default=None)
    range_experiment.add_argument("--workload-dir", type=Path, default=None)
    range_experiment.add_argument("--results-dir", type=Path, default=None)
    range_experiment.add_argument("--query-count", type=int, default=None)
    range_experiment.add_argument("--seed", type=int, default=None)
    range_experiment.add_argument("--train-fraction", type=float, default=None)
    range_experiment.add_argument("--k-candidates", default=None)
    range_experiment.add_argument("--delta-candidates", default=None)
    range_experiment.add_argument("--folds", type=int, default=None)

    shift_experiment = subparsers.add_parser(
        "run-distribution-shift",
        help="Evaluate learned indexes before and after controlled distribution shifts.",
    )
    shift_experiment.add_argument("--data-dir", type=Path, default=None)
    shift_experiment.add_argument("--results-dir", type=Path, default=None)
    shift_experiment.add_argument("--point-query-count", type=int, default=None)
    shift_experiment.add_argument("--range-query-count", type=int, default=None)
    shift_experiment.add_argument("--train-fraction", type=float, default=None)
    shift_experiment.add_argument("--seed", type=int, default=None)
    shift_experiment.add_argument("--shift-fractions", default="0.05,0.10,0.20")
    shift_experiment.add_argument("--k-candidates", default=None)
    shift_experiment.add_argument("--delta-candidates", default=None)
    shift_experiment.add_argument("--folds", type=int, default=None)

    return parser


def extract_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    settings = Neo4jSettings.from_env()
    database = args.database if args.database is not None else settings.database
    properties = args.properties or config.data.properties
    output_dir = args.output_dir or config.data.output_dir

    for property_name in properties:
        pairs = extract_property_pairs(settings, property_name, database=database)
        records = preprocess_pairs(pairs)
        if not records:
            raise EmptyExtractionError(
                f"No records found for {property_name!r}. "
                "Check that your Neo4j database is the Recommendations dataset and that the "
                "property exists on Movie nodes."
            )
        output_path = output_dir / f"{property_name}.csv"
        write_records_csv(output_path, records)
        print(f"Wrote {len(records)} records to {output_path}")

    return 0


def create_indexes_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    settings = Neo4jSettings.from_env()
    database = args.database if args.database is not None else settings.database
    properties = args.properties or config.data.properties

    try:
        index_names = create_neo4j_range_indexes(settings, properties, database=database)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for index_name in index_names:
        print(f"Created or verified Neo4j range index: {index_name}")
    return 0


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def query_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    records = read_records_csv(args.csv_path)
    rmi_tuning = None
    rmi_k = args.rmi_k if args.rmi_k is not None else config.rmi.k
    rmi_delta = args.rmi_delta if args.rmi_delta is not None else config.rmi.delta
    btree_order = args.order if args.order is not None else config.btree.order
    pgm_epsilon = args.pgm_epsilon if args.pgm_epsilon is not None else config.pgm.epsilon

    if args.index == "rmi" and args.tune_rmi:
        rmi_tuning = tune_rmi(
            records,
            k_candidates=parse_int_list(args.rmi_k_candidates) or config.rmi.tuning.k_candidates,
            delta_candidates=parse_int_list(args.rmi_delta_candidates)
            if args.rmi_delta_candidates is not None
            else config.rmi.tuning.delta_candidates or None,
            folds=args.rmi_folds or config.rmi.tuning.folds,
        )
        rmi_k = rmi_tuning.k
        rmi_delta = rmi_tuning.delta

    if args.index == "btree":
        index = BTreeIndex(records, order=btree_order)
    elif args.index == "pgm":
        index = StaticPGMIndex(records, epsilon=pgm_epsilon)
    elif args.index == "rmi":
        index = RMIIndex(records, k=rmi_k, delta=rmi_delta)
    else:
        index = SortedArrayIndex(records)

    if args.exact is not None:
        matches = index.exact(args.exact)
    else:
        matches = index.range(args.minimum, args.maximum)

    displayed_matches = matches[: args.limit] if args.limit is not None else matches

    for record in displayed_matches:
        print(f"{record.position}\t{record.value}\t{record.node_id}")

    if args.limit is not None and len(matches) > args.limit:
        print(f"Displayed {args.limit} of {len(matches)} matches")

    print(f"Matched {len(matches)} of {len(index)} records")
    print(f"Index: {args.index}")
    if args.index == "pgm":
        print(f"PGM epsilon: {index.epsilon}")
    if args.index == "rmi":
        print(f"RMI k: {index.k}")
        print(f"RMI delta: {index.delta}")
        if rmi_tuning is not None:
            print(f"RMI CV mean abs error: {rmi_tuning.mean_abs_error:.2f}")
            print(f"RMI CV max abs error: {rmi_tuning.max_abs_error}")
            print(f"RMI CV coverage: {rmi_tuning.coverage:.3f}")
    return 0


def tune_rmi_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    records = read_records_csv(args.csv_path)
    result = tune_rmi(
        records,
        k_candidates=parse_int_list(args.k_candidates) or config.rmi.tuning.k_candidates,
        delta_candidates=parse_int_list(args.delta_candidates)
        if args.delta_candidates is not None
        else config.rmi.tuning.delta_candidates or None,
        folds=args.folds or config.rmi.tuning.folds,
    )
    print(f"Best RMI k: {result.k}")
    print(f"Best RMI delta: {result.delta}")
    print(f"Mean absolute prediction error: {result.mean_abs_error:.2f}")
    print(f"Max absolute prediction error: {result.max_abs_error}")
    print(f"Delta coverage: {result.coverage:.3f}")
    return 0


def generate_workload_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    paths = generate_workloads(
        data_dir=args.data_dir or config.data.output_dir,
        workload_dir=args.workload_dir or config.experiment.workload_dir,
        properties=config.data.properties,
        train_fraction=args.train_fraction or config.experiment.train_fraction,
        query_count=args.query_count or config.experiment.query_count,
        seed=args.seed if args.seed is not None else config.experiment.seed,
    )
    for property_name, path in paths.items():
        print(f"Wrote {property_name} point queries to {path}")
    return 0


def generate_range_workload_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    paths = generate_range_workloads(
        data_dir=args.data_dir or config.data.output_dir,
        workload_dir=args.workload_dir or config.experiment.workload_dir,
        properties=config.data.properties,
        train_fraction=args.train_fraction or config.experiment.train_fraction,
        query_count=args.query_count or config.experiment.query_count,
        seed=args.seed if args.seed is not None else config.experiment.seed,
    )
    for property_name, path in paths.items():
        print(f"Wrote {property_name} range queries to {path}")
    return 0


def run_experiment_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    paths = run_experiments(
        data_dir=args.data_dir or config.data.output_dir,
        workload_dir=args.workload_dir or config.experiment.workload_dir,
        results_dir=args.results_dir or config.experiment.results_dir,
        properties=config.data.properties,
        train_fraction=args.train_fraction or config.experiment.train_fraction,
        query_count=args.query_count or config.experiment.query_count,
        seed=args.seed if args.seed is not None else config.experiment.seed,
        btree_order=config.btree.order,
        pgm_epsilon_candidates=config.pgm.tuning.epsilon_candidates,
        k_candidates=parse_int_list(args.k_candidates) or config.rmi.tuning.k_candidates,
        delta_candidates=parse_int_list(args.delta_candidates)
        or config.rmi.tuning.delta_candidates,
        folds=args.folds or config.rmi.tuning.folds,
    )
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    return 0


def run_range_experiment_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    paths = run_range_experiments(
        data_dir=args.data_dir or config.data.output_dir,
        workload_dir=args.workload_dir or config.experiment.workload_dir,
        results_dir=args.results_dir or config.experiment.results_dir,
        properties=config.data.properties,
        train_fraction=args.train_fraction or config.experiment.train_fraction,
        query_count=args.query_count or config.experiment.query_count,
        seed=args.seed if args.seed is not None else config.experiment.seed,
        btree_order=config.btree.order,
        pgm_epsilon_candidates=config.pgm.tuning.epsilon_candidates,
        k_candidates=parse_int_list(args.k_candidates) or config.rmi.tuning.k_candidates,
        delta_candidates=parse_int_list(args.delta_candidates)
        or config.rmi.tuning.delta_candidates,
        folds=args.folds or config.rmi.tuning.folds,
    )
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    return 0


def run_distribution_shift_command(args: argparse.Namespace) -> int:
    config = ExperimentSettings.from_file(args.config)
    paths = run_distribution_shift_experiment(
        data_dir=args.data_dir or config.data.output_dir,
        results_dir=args.results_dir or config.experiment.results_dir,
        properties=config.data.properties,
        train_fraction=args.train_fraction or config.experiment.train_fraction,
        point_query_count=args.point_query_count or config.experiment.query_count,
        range_query_count=args.range_query_count or config.experiment.query_count,
        shift_fractions=parse_float_list(args.shift_fractions) or [0.05, 0.10, 0.20],
        seed=args.seed if args.seed is not None else config.experiment.seed,
        btree_order=config.btree.order,
        pgm_epsilon_candidates=config.pgm.tuning.epsilon_candidates,
        k_candidates=parse_int_list(args.k_candidates) or config.rmi.tuning.k_candidates,
        delta_candidates=parse_int_list(args.delta_candidates)
        or config.rmi.tuning.delta_candidates,
        folds=args.folds or config.rmi.tuning.folds,
    )
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        return extract_command(args)
    if args.command == "create-indexes":
        return create_indexes_command(args)
    if args.command == "query":
        return query_command(args)
    if args.command == "tune-rmi":
        return tune_rmi_command(args)
    if args.command == "generate-workload":
        return generate_workload_command(args)
    if args.command == "generate-range-workload":
        return generate_range_workload_command(args)
    if args.command == "run-experiment":
        return run_experiment_command(args)
    if args.command == "run-range-experiment":
        return run_range_experiment_command(args)
    if args.command == "run-distribution-shift":
        return run_distribution_shift_command(args)

    parser.print_help()
    return 0
