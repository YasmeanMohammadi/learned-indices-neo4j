import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from learned_indices_neo4j.experiments import (
    generate_range_workloads,
    generate_workloads,
    read_point_queries,
    read_range_queries,
    run_experiments,
    run_range_experiments,
    split_records,
    tune_rmi_grid,
)
from learned_indices_neo4j.io import write_records_csv
from learned_indices_neo4j.records import preprocess_pairs


class TestExperiments(unittest.TestCase):
    def setUp(self):
        self.records = preprocess_pairs(
            [
                (1990, "movie-1"),
                (1991, "movie-2"),
                (1992, "movie-3"),
                (1993, "movie-4"),
                (1994, "movie-5"),
                (1995, "movie-6"),
                (1996, "movie-7"),
                (1997, "movie-8"),
                (1998, "movie-9"),
                (1999, "movie-10"),
            ]
        )

    def test_split_records_keeps_train_and_test_sizes(self):
        split = split_records(self.records, train_fraction=0.8, seed=1)

        self.assertEqual(len(split.train), 8)
        self.assertEqual(len(split.test), 2)

    def test_generate_workloads_writes_point_query_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            workload_dir = root / "workloads"
            write_records_csv(data_dir / "year.csv", self.records)

            paths = generate_workloads(
                data_dir=data_dir,
                workload_dir=workload_dir,
                properties=["year"],
                train_fraction=0.8,
                query_count=2,
                seed=1,
            )

            queries = read_point_queries(paths["year"])

        self.assertEqual(len(queries), 2)
        self.assertEqual({query.property_name for query in queries}, {"year"})

    def test_generate_range_workloads_writes_range_query_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            workload_dir = root / "workloads"
            write_records_csv(data_dir / "year.csv", self.records)

            paths = generate_range_workloads(
                data_dir=data_dir,
                workload_dir=workload_dir,
                properties=["year"],
                train_fraction=0.8,
                query_count=2,
                seed=1,
            )

            queries = read_range_queries(paths["year"])

        self.assertEqual(len(queries), 2)
        self.assertEqual({query.property_name for query in queries}, {"year"})
        self.assertTrue(all(query.true_result_count >= 1 for query in queries))

    def test_tune_rmi_grid_returns_best_config_and_sensitivity_rows(self):
        split = split_records(self.records, train_fraction=0.8, seed=1)

        tuned, rows = tune_rmi_grid(
            property_name="year",
            records=self.records,
            train_records=split.train,
            k_candidates=[1, 2],
            delta_candidates=[1, 4],
            folds=2,
        )

        self.assertIn(tuned.k, {1, 2})
        self.assertIn(tuned.delta, {1, 4})
        self.assertEqual(len(rows), 4)

    def test_run_experiments_writes_report_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            workload_dir = root / "workloads"
            results_dir = root / "results"
            write_records_csv(data_dir / "year.csv", self.records)

            paths = run_experiments(
                data_dir=data_dir,
                workload_dir=workload_dir,
                results_dir=results_dir,
                properties=["year"],
                train_fraction=0.8,
                query_count=2,
                seed=1,
                btree_order=3,
                pgm_epsilon_candidates=[2, 4],
                k_candidates=[1, 2],
                delta_candidates=[1, 4],
                folds=2,
            )

            main_table = paths["main_comparison"].read_text(encoding="utf-8")
            sensitivity = paths["rmi_hyperparameter_sensitivity"].read_text(
                encoding="utf-8"
            )
            pgm_sensitivity = paths["pgm_hyperparameter_sensitivity"].read_text(
                encoding="utf-8"
            )
            worst_cases = paths["rmi_worst_case_queries"].read_text(encoding="utf-8")
            lookup_latency = paths["lookup_latency_detail"].read_text(encoding="utf-8")

        self.assertIn("MAE (positions)", main_table)
        self.assertIn("Index build time (ms)", main_table)
        self.assertIn("Evaluation execution time (ms)", main_table)
        self.assertIn("Lookup latency (ms, avg)", main_table)
        self.assertIn("RMI", main_table)
        self.assertIn("PGM-static", main_table)
        self.assertIn("validation_coverage", sensitivity)
        self.assertIn("epsilon", pgm_sensitivity)
        self.assertIn("absolute_error", worst_cases)
        self.assertIn("lookup_latency_ms", lookup_latency)
        self.assertIn("B-Tree", lookup_latency)
        self.assertIn("RMI", lookup_latency)
        self.assertIn("PGM-static", lookup_latency)

    def test_run_range_experiments_writes_report_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            workload_dir = root / "workloads"
            results_dir = root / "results"
            write_records_csv(data_dir / "year.csv", self.records)

            paths = run_range_experiments(
                data_dir=data_dir,
                workload_dir=workload_dir,
                results_dir=results_dir,
                properties=["year"],
                train_fraction=0.8,
                query_count=2,
                seed=1,
                btree_order=3,
                pgm_epsilon_candidates=[2, 4],
                k_candidates=[1, 2],
                delta_candidates=[1, 4],
                folds=2,
            )

            main_table = paths["range_main_comparison"].read_text(encoding="utf-8")
            worst_cases = paths["range_worst_case_queries"].read_text(encoding="utf-8")
            lookup_latency = paths["range_lookup_latency_detail"].read_text(
                encoding="utf-8"
            )

        self.assertIn("Start boundary MAE (positions)", main_table)
        self.assertIn("Exact range correctness", main_table)
        self.assertIn("PGM epsilon", main_table)
        self.assertIn("result_count_error", worst_cases)
        self.assertIn("binary_search_comparisons", lookup_latency)
        self.assertIn("B-Tree", lookup_latency)
        self.assertIn("RMI", lookup_latency)
        self.assertIn("PGM-static", lookup_latency)


if __name__ == "__main__":
    unittest.main()
