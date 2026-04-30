import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from learned_indices_neo4j.distribution_shift import (
    generate_shift_scenarios,
    run_distribution_shift_experiment,
)
from learned_indices_neo4j.io import write_records_csv
from learned_indices_neo4j.records import preprocess_pairs


class TestDistributionShift(unittest.TestCase):
    def setUp(self):
        self.year_records = preprocess_pairs(
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

    def test_generate_shift_scenarios_creates_inserted_records(self):
        scenarios = generate_shift_scenarios(
            "year",
            self.year_records,
            shift_fractions=[0.2],
            seed=1,
        )

        self.assertEqual({scenario.scenario_name for scenario in scenarios}, {"recent_tail", "duplicate_burst"})
        self.assertTrue(all(scenario.inserted_records for scenario in scenarios))
        self.assertTrue(all(len(scenario.shifted_records) > len(self.year_records) for scenario in scenarios))

    def test_run_distribution_shift_experiment_writes_summary_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            results_dir = root / "results"
            write_records_csv(data_dir / "year.csv", self.year_records)

            paths = run_distribution_shift_experiment(
                data_dir=data_dir,
                results_dir=results_dir,
                properties=["year"],
                train_fraction=0.8,
                point_query_count=2,
                range_query_count=2,
                shift_fractions=[0.2],
                seed=1,
                btree_order=3,
                pgm_epsilon_candidates=[2, 4],
                k_candidates=[1, 2],
                delta_candidates=[1, 4],
                folds=2,
            )

            summary = paths["distribution_shift_summary"].read_text(encoding="utf-8")
            point = paths["distribution_shift_point"].read_text(encoding="utf-8")
            range_output = paths["distribution_shift_range"].read_text(encoding="utf-8")

        self.assertIn("shifted_stale_models", summary)
        self.assertIn("shifted_retrained_models", summary)
        self.assertIn("baseline", summary)
        self.assertIn("point", point)
        self.assertIn("range", range_output)
        self.assertIn("exact_correctness", summary)


if __name__ == "__main__":
    unittest.main()
