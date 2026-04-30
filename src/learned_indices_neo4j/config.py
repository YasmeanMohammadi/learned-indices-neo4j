from dataclasses import dataclass
from os import environ, getenv
from pathlib import Path
import tomllib


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


@dataclass(frozen=True)
class DataSettings:
    output_dir: Path
    properties: list[str]


@dataclass(frozen=True)
class BTreeSettings:
    order: int


@dataclass(frozen=True)
class PGMTuningSettings:
    epsilon_candidates: list[int]


@dataclass(frozen=True)
class PGMSettings:
    epsilon: int
    tuning: PGMTuningSettings


@dataclass(frozen=True)
class RMITuningSettings:
    k_candidates: list[int]
    delta_candidates: list[int]
    folds: int


@dataclass(frozen=True)
class RMISettings:
    k: int
    delta: int | None
    auto_delta: bool
    tuning: RMITuningSettings


@dataclass(frozen=True)
class ExperimentRunSettings:
    seed: int
    train_fraction: float
    query_count: int
    workload_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class ExperimentSettings:
    data: DataSettings
    btree: BTreeSettings
    pgm: PGMSettings
    rmi: RMISettings
    experiment: ExperimentRunSettings

    @classmethod
    def from_file(cls, path: str | Path = "experiment.toml") -> "ExperimentSettings":
        config_path = Path(path)
        raw = {}
        if config_path.exists():
            with config_path.open("rb") as file:
                raw = tomllib.load(file)

        data = raw.get("data", {})
        btree = raw.get("btree", {})
        pgm = raw.get("pgm", {})
        pgm_tuning = pgm.get("tuning", {})
        rmi = raw.get("rmi", {})
        tuning = rmi.get("tuning", {})
        experiment = raw.get("experiment", {})
        auto_delta = bool(rmi.get("auto_delta", True))
        delta = rmi.get("delta")

        return cls(
            data=DataSettings(
                output_dir=Path(data.get("output_dir", "data")),
                properties=list(data.get("properties", ["year", "imdbVotes"])),
            ),
            btree=BTreeSettings(order=int(btree.get("order", 64))),
            pgm=PGMSettings(
                epsilon=int(pgm.get("epsilon", 64)),
                tuning=PGMTuningSettings(
                    epsilon_candidates=[
                        int(value)
                        for value in pgm_tuning.get("epsilon_candidates", [8, 16, 32, 64, 128])
                    ]
                ),
            ),
            rmi=RMISettings(
                k=int(rmi.get("k", 16)),
                delta=None if auto_delta else int(delta if delta is not None else 0),
                auto_delta=auto_delta,
                tuning=RMITuningSettings(
                    k_candidates=[int(value) for value in tuning.get("k_candidates", [5, 10, 20, 50])],
                    delta_candidates=[
                        int(value) for value in tuning.get("delta_candidates", [10, 25, 50, 100])
                    ],
                    folds=int(tuning.get("folds", 5)),
                ),
            ),
            experiment=ExperimentRunSettings(
                seed=int(experiment.get("seed", 42)),
                train_fraction=float(experiment.get("train_fraction", 0.8)),
                query_count=int(experiment.get("query_count", 200)),
                workload_dir=Path(experiment.get("workload_dir", "workloads")),
                results_dir=Path(experiment.get("results_dir", "results")),
            ),
        )
