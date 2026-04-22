from collections.abc import Iterable

from learned_indices_neo4j.config import Neo4jSettings


class EmptyExtractionError(RuntimeError):
    pass


PROPERTY_SPECS = {
    "year": ("Movie", "year"),
    "imdbVotes": ("Movie", "imdbVotes"),
}


PROPERTY_QUERIES: dict[str, str] = {
    "year": """
        MATCH (movie:Movie)
        WHERE movie.year IS NOT NULL
        RETURN movie.year AS value, elementId(movie) AS node_id
    """,
    "imdbVotes": """
        MATCH (movie:Movie)
        WHERE movie.imdbVotes IS NOT NULL
        RETURN movie.imdbVotes AS value, elementId(movie) AS node_id
    """,
}


def _validate_property_name(property_name: str) -> None:
    if property_name not in PROPERTY_QUERIES:
        valid = ", ".join(sorted(PROPERTY_QUERIES))
        raise ValueError(f"Unknown property {property_name!r}. Valid properties: {valid}")


def extract_property_pairs(
    settings: Neo4jSettings,
    property_name: str,
    *,
    database: str | None = None,
) -> list[tuple[object, object]]:
    _validate_property_name(property_name)

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "The Neo4j driver is not installed. Run `python -m pip install -e .` first."
        ) from exc

    driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
    try:
        with driver.session(database=database) as session:
            rows: Iterable[dict[str, object]] = session.run(PROPERTY_QUERIES[property_name])
            return [(row["value"], row["node_id"]) for row in rows]
    finally:
        driver.close()


def create_neo4j_range_indexes(
    settings: Neo4jSettings,
    property_names: list[str],
    *,
    database: str | None = None,
) -> list[str]:
    for property_name in property_names:
        _validate_property_name(property_name)

    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import Forbidden
    except ImportError as exc:
        raise RuntimeError(
            "The Neo4j driver is not installed. Run `python -m pip install -e .` first."
        ) from exc

    driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
    created_index_names: list[str] = []
    try:
        with driver.session(database=database) as session:
            for property_name in property_names:
                label, property_key = PROPERTY_SPECS[property_name]
                index_name = f"movie_{property_key}_range_idx"
                session.run(
                    f"CREATE INDEX {index_name} IF NOT EXISTS "
                    f"FOR (node:{label}) ON (node.{property_key})"
                ).consume()
                created_index_names.append(index_name)
        return created_index_names
    except Forbidden as exc:
        raise PermissionError(
            "Neo4j refused to create schema indexes for this user. "
            "The Recommendations sandbox user can read data, but it does not have "
            "`create_index` permission. The local sorted-array baseline and RMI CSV extraction "
            "can still run."
        ) from exc
    finally:
        driver.close()
