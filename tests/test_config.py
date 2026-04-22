import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from learned_indices_neo4j.config import Neo4jSettings


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


if __name__ == "__main__":
    unittest.main()
