from dataclasses import dataclass
from os import environ, getenv
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in environ:
            environ[key] = value


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str | None = None

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Neo4jSettings":
        if env_file is not None:
            load_env_file(env_file)

        return cls(
            uri=getenv("NEO4J_URI", "neo4j://localhost:7687"),
            user=getenv("NEO4J_USER", "neo4j"),
            password=getenv("NEO4J_PASSWORD", "password"),
            database=getenv("NEO4J_DATABASE"),
        )
