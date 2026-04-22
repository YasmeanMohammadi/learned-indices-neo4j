import argparse
import sys
from pathlib import Path

from learned_indices_neo4j.btree_index import BTreeIndex
from learned_indices_neo4j.config import Neo4jSettings
from learned_indices_neo4j.io import read_records_csv, write_records_csv
from learned_indices_neo4j.neo4j_extractor import (
    EmptyExtractionError,
    PROPERTY_QUERIES,
    create_neo4j_range_indexes,
    extract_property_pairs,
)
from learned_indices_neo4j.records import preprocess_pairs
from learned_indices_neo4j.rmi_index import RMIIndex, tune_rmi
from learned_indices_neo4j.sorted_array_index import SortedArrayIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learned-indices-neo4j",
        description="Build baseline sorted-array indexes from Neo4j Recommendations properties.",
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
        default=Path("data"),
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
        choices=["sorted-array", "btree", "rmi"],
        default="sorted-array",
        help="Local index implementation to use for CSV lookups.",
    )
    query.add_argument(
        "--order",
        type=int,
        default=64,
        help="B-tree order when --index btree is selected.",
    )
    query.add_argument("--rmi-k", type=int, default=16, help="Number of stage-2 RMI models.")
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
        default="4,8,16,32",
        help="Comma-separated k candidates used by --tune-rmi.",
    )
    query.add_argument(
        "--rmi-delta-candidates",
        default=None,
        help="Optional comma-separated delta candidates used by --tune-rmi.",
    )
    query.add_argument("--rmi-folds", type=int, default=5, help="Cross-validation folds.")

    tune = subparsers.add_parser(
        "tune-rmi",
        help="Cross-validate RMI k and delta for a preprocessed CSV file.",
    )
    tune.add_argument("csv_path", type=Path)
    tune.add_argument("--k-candidates", default="4,8,16,32")
    tune.add_argument("--delta-candidates", default=None)
    tune.add_argument("--folds", type=int, default=5)

    return parser


def extract_command(args: argparse.Namespace) -> int:
    settings = Neo4jSettings.from_env()
    database = args.database if args.database is not None else settings.database
    properties = args.properties or sorted(PROPERTY_QUERIES)

    for property_name in properties:
        pairs = extract_property_pairs(settings, property_name, database=database)
        records = preprocess_pairs(pairs)
        if not records:
            raise EmptyExtractionError(
                f"No records found for {property_name!r}. "
                "Check that your Neo4j database is the Recommendations dataset and that the "
                "property exists on Movie nodes."
            )
        output_path = args.output_dir / f"{property_name}.csv"
        write_records_csv(output_path, records)
        print(f"Wrote {len(records)} records to {output_path}")

    return 0


def create_indexes_command(args: argparse.Namespace) -> int:
    settings = Neo4jSettings.from_env()
    database = args.database if args.database is not None else settings.database
    properties = args.properties or sorted(PROPERTY_QUERIES)

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


def query_command(args: argparse.Namespace) -> int:
    records = read_records_csv(args.csv_path)
    rmi_tuning = None
    if args.index == "rmi" and args.tune_rmi:
        rmi_tuning = tune_rmi(
            records,
            k_candidates=parse_int_list(args.rmi_k_candidates) or [16],
            delta_candidates=parse_int_list(args.rmi_delta_candidates),
            folds=args.rmi_folds,
        )
        args.rmi_k = rmi_tuning.k
        args.rmi_delta = rmi_tuning.delta

    if args.index == "btree":
        index = BTreeIndex(records, order=args.order)
    elif args.index == "rmi":
        index = RMIIndex(records, k=args.rmi_k, delta=args.rmi_delta)
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
    if args.index == "rmi":
        print(f"RMI k: {index.k}")
        print(f"RMI delta: {index.delta}")
        if rmi_tuning is not None:
            print(f"RMI CV mean abs error: {rmi_tuning.mean_abs_error:.2f}")
            print(f"RMI CV max abs error: {rmi_tuning.max_abs_error}")
            print(f"RMI CV coverage: {rmi_tuning.coverage:.3f}")
    return 0


def tune_rmi_command(args: argparse.Namespace) -> int:
    records = read_records_csv(args.csv_path)
    result = tune_rmi(
        records,
        k_candidates=parse_int_list(args.k_candidates) or [16],
        delta_candidates=parse_int_list(args.delta_candidates),
        folds=args.folds,
    )
    print(f"Best RMI k: {result.k}")
    print(f"Best RMI delta: {result.delta}")
    print(f"Mean absolute prediction error: {result.mean_abs_error:.2f}")
    print(f"Max absolute prediction error: {result.max_abs_error}")
    print(f"Delta coverage: {result.coverage:.3f}")
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

    parser.print_help()
    return 0
