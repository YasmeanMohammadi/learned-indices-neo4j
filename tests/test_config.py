import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from learned_indices_neo4j.config import ExperimentSettings, Neo4jSettings


class TestNeo4jSettings(unittest.TestCase):
    def test_settings_have_local_defaults(self):
        old_values = {
            "NEO4J_URI": os.environ.pop("NEO4J_URI", None),
            "NEO4J_USER": os.environ.pop("NEO4J_USER", None),
            "NEO4J_PASSWORD": os.environ.pop("NEO4J_PASSWORD", None),
            "NEO4J_DATABASE": os.environ.pop("NEO4J_DATABASE", None),
        }

        try:
            settings = Neo4jSettings.from_env(env_file=None)
        finally:
            for key, value in old_values.items():
                if value is not None:
                    os.environ[key] = value

        self.assertEqual(settings.uri, "neo4j://localhost:7687")
        self.assertEqual(settings.user, "neo4j")
        self.assertEqual(settings.password, "password")
        self.assertIsNone(settings.database)


class TestExperimentSettings(unittest.TestCase):
    def test_missing_config_uses_defaults(self):
        settings = ExperimentSettings.from_file("missing-experiment-config.toml")

        self.assertEqual(settings.data.output_dir, Path("data"))
        self.assertEqual(settings.data.properties, ["year", "imdbVotes"])
        self.assertEqual(settings.btree.order, 64)
        self.assertEqual(settings.rmi.k, 16)
        self.assertIsNone(settings.rmi.delta)
        self.assertTrue(settings.rmi.auto_delta)
        self.assertEqual(settings.rmi.tuning.k_candidates, [5, 10, 20, 50])
        self.assertEqual(settings.rmi.tuning.delta_candidates, [10, 25, 50, 100])
        self.assertEqual(settings.rmi.tuning.folds, 5)
        self.assertEqual(settings.experiment.seed, 42)
        self.assertEqual(settings.experiment.train_fraction, 0.8)
        self.assertEqual(settings.experiment.query_count, 200)
        self.assertEqual(settings.experiment.workload_dir, Path("workloads"))
        self.assertEqual(settings.experiment.results_dir, Path("results"))

    def test_config_file_overrides_hyperparameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment.toml"
            path.write_text(
                """
[data]
output_dir = "tmp-data"
properties = ["year"]

[btree]
order = 32

[rmi]
k = 8
delta = 12
auto_delta = false

[rmi.tuning]
k_candidates = [2, 4, 8]
delta_candidates = [4, 8, 12]
folds = 3

[experiment]
seed = 7
train_fraction = 0.75
query_count = 25
workload_dir = "tmp-workloads"
results_dir = "tmp-results"
""".strip(),
                encoding="utf-8",
            )

            settings = ExperimentSettings.from_file(path)

        self.assertEqual(settings.data.output_dir, Path("tmp-data"))
        self.assertEqual(settings.data.properties, ["year"])
        self.assertEqual(settings.btree.order, 32)
        self.assertEqual(settings.rmi.k, 8)
        self.assertEqual(settings.rmi.delta, 12)
        self.assertFalse(settings.rmi.auto_delta)
        self.assertEqual(settings.rmi.tuning.k_candidates, [2, 4, 8])
        self.assertEqual(settings.rmi.tuning.delta_candidates, [4, 8, 12])
        self.assertEqual(settings.rmi.tuning.folds, 3)
        self.assertEqual(settings.experiment.seed, 7)
        self.assertEqual(settings.experiment.train_fraction, 0.75)
        self.assertEqual(settings.experiment.query_count, 25)
        self.assertEqual(settings.experiment.workload_dir, Path("tmp-workloads"))
        self.assertEqual(settings.experiment.results_dir, Path("tmp-results"))


if __name__ == "__main__":
    unittest.main()
