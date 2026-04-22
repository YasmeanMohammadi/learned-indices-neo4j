import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from learned_indices_neo4j.btree_index import BTreeIndex
from learned_indices_neo4j.io import read_records_csv, write_records_csv
from learned_indices_neo4j.neo4j_extractor import PROPERTY_QUERIES, PROPERTY_SPECS
from learned_indices_neo4j.records import preprocess_pairs
from learned_indices_neo4j.rmi_index import LinearModel, RMIIndex, tune_rmi
from learned_indices_neo4j.sorted_array_index import SortedArrayIndex


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_removes_nulls_sorts_and_assigns_positions(self):
        records = preprocess_pairs(
            [
                (1999, "movie-3"),
                (None, "movie-4"),
                (1972, "movie-1"),
                (1999, "movie-2"),
            ]
        )

        self.assertEqual(
            [(record.value, record.node_id, record.position) for record in records],
            [
                (1972, "movie-1", 0),
                (1999, "movie-2", 1),
                (1999, "movie-3", 2),
            ],
        )


class TestSortedArrayIndex(unittest.TestCase):
    def setUp(self):
        self.index = SortedArrayIndex.from_pairs(
            [
                (1960, "person-1"),
                (1980, "person-3"),
                (1980, "person-2"),
                (1990, "person-4"),
            ]
        )

    def test_exact_lookup_uses_binary_search_bounds(self):
        matches = self.index.exact(1980)

        self.assertEqual([record.node_id for record in matches], ["person-2", "person-3"])
        self.assertEqual([record.position for record in matches], [1, 2])

    def test_range_lookup_returns_inclusive_window(self):
        matches = self.index.range(1970, 1980)

        self.assertEqual([record.node_id for record in matches], ["person-2", "person-3"])

    def test_estimate_position_returns_insertion_position(self):
        self.assertEqual(self.index.estimate_position(1970), 1)
        self.assertEqual(self.index.estimate_position(2000), 4)


class TestBTreeIndex(unittest.TestCase):
    def setUp(self):
        self.index = BTreeIndex.from_pairs(
            [
                (1960, "person-1"),
                (1980, "person-3"),
                (1980, "person-2"),
                (1990, "person-4"),
                (2000, "person-5"),
            ],
            order=3,
        )

    def test_exact_lookup_returns_duplicate_keys(self):
        matches = self.index.exact(1980)

        self.assertEqual([record.node_id for record in matches], ["person-2", "person-3"])
        self.assertEqual([record.position for record in matches], [1, 2])

    def test_range_lookup_scans_linked_leaves(self):
        matches = self.index.range(1970, 1990)

        self.assertEqual(
            [(record.value, record.node_id) for record in matches],
            [
                (1980, "person-2"),
                (1980, "person-3"),
                (1990, "person-4"),
            ],
        )

    def test_exclusive_range_bounds(self):
        matches = self.index.range(1960, 2000, include_minimum=False, include_maximum=False)

        self.assertEqual([record.value for record in matches], [1980, 1980, 1990])

    def test_tree_has_multiple_levels_for_small_order(self):
        self.assertGreaterEqual(self.index.height(), 2)

    def test_exact_lookup_finds_duplicates_split_across_leaves(self):
        index = BTreeIndex.from_pairs(
            [
                (1994, "movie-1"),
                (1995, "movie-2"),
                (1995, "movie-3"),
                (1995, "movie-4"),
                (1995, "movie-5"),
                (1996, "movie-6"),
            ],
            order=3,
        )

        matches = index.exact(1995)

        self.assertEqual(
            [record.node_id for record in matches],
            ["movie-2", "movie-3", "movie-4", "movie-5"],
        )


class TestRMIIndex(unittest.TestCase):
    def setUp(self):
        self.records = preprocess_pairs(
            [
                (1994, "movie-1"),
                (1995, "movie-2"),
                (1995, "movie-3"),
                (1995, "movie-4"),
                (1996, "movie-5"),
                (1997, "movie-6"),
                (2001, "movie-7"),
            ]
        )
        self.index = RMIIndex(self.records, k=2)

    def test_linear_model_fits_simple_position_curve(self):
        model = LinearModel.fit(preprocess_pairs([(10, "a"), (20, "b"), (30, "c")]))

        self.assertEqual(round(model.predict(20)), 1)

    def test_exact_lookup_uses_prediction_window_and_expands_duplicates(self):
        matches = self.index.exact(1995)

        self.assertEqual([record.node_id for record in matches], ["movie-2", "movie-3", "movie-4"])

    def test_exact_binary_lookup_uses_prediction_window_and_expands_duplicates(self):
        matches = self.index.exact_binary(1995)

        self.assertEqual([record.node_id for record in matches], ["movie-2", "movie-3", "movie-4"])
        self.assertGreaterEqual(self.index.binary_search_comparisons(1995), 1)

    def test_range_lookup_returns_expected_window(self):
        matches = self.index.range(1995, 1997)

        self.assertEqual(
            [(record.value, record.node_id) for record in matches],
            [
                (1995, "movie-2"),
                (1995, "movie-3"),
                (1995, "movie-4"),
                (1996, "movie-5"),
                (1997, "movie-6"),
            ],
        )

    def test_tune_rmi_selects_candidate_k_and_delta(self):
        result = tune_rmi(
            self.records,
            k_candidates=[1, 2, 4],
            delta_candidates=[0, 1, 2, 4],
            folds=3,
        )

        self.assertIn(result.k, {1, 2, 4})
        self.assertIn(result.delta, {0, 1, 2, 4})
        self.assertGreaterEqual(result.coverage, 0)
        self.assertLessEqual(result.coverage, 1)


class TestCsvIo(unittest.TestCase):
    def test_records_round_trip_to_csv(self):
        records = preprocess_pairs([(1995, "movie-2"), (1990, "movie-1")])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "released.csv"
            write_records_csv(path, records)

            loaded = read_records_csv(path)

        self.assertEqual(loaded, records)


class TestRecommendationsExtractors(unittest.TestCase):
    def test_default_properties_target_recommendations_movie_fields(self):
        self.assertEqual(set(PROPERTY_QUERIES), {"year", "imdbVotes"})
        self.assertIn("movie.year", PROPERTY_QUERIES["year"])
        self.assertIn("movie.imdbVotes", PROPERTY_QUERIES["imdbVotes"])
        self.assertEqual(PROPERTY_SPECS["year"], ("Movie", "year"))
        self.assertEqual(PROPERTY_SPECS["imdbVotes"], ("Movie", "imdbVotes"))


if __name__ == "__main__":
    unittest.main()
