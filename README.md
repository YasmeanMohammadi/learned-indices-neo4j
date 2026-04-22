# learned indices neo4j

A Python project for experimenting with learned indexing ideas on graph data in Neo4j.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Copy `.env.example` to `.env` and update the Neo4j connection values.

## Experiment Plan

This first version builds the preprocessing output for the Neo4j Recommendations database:

- `Movie.year`
- `Movie.imdbVotes`

Each property is extracted as `(property_value, node_id)`, null values are removed, records are
sorted by property value, and each sorted record receives a positional index from `0` to `n - 1`.

The local baselines include:

- a sorted-array binary-search index
- a bulk-loaded B+ tree index with linked leaves for range scans

The learned index is a two-stage recursive model index:

- stage 1: one linear regression model trained on all `(value, position)` pairs
- stage 2: `k` independent linear regression models selected by the stage-1 prediction
- lookup: stage 1 chooses a partition, stage 2 predicts a position, and the index scans a local
  `delta` window around that prediction
- tuning: `k` and `delta` can be selected with cross-validation

Neo4j's native range indexes can also be created for the same fields. These are the database-side
baseline indexes for Cypher query planning.

## Run

Extract and preprocess both properties:

```bash
python -m learned_indices_neo4j extract --output-dir data
```

Create Neo4j native range indexes for both properties:

```bash
python -m learned_indices_neo4j create-indexes
```

Some hosted Recommendations sandbox users are read-only for schema operations. If Neo4j refuses
`create-indexes`, continue with `extract`; the local sorted-array baseline and RMI data still work.

Extract only one property:

```bash
python -m learned_indices_neo4j extract --property year --output-dir data
python -m learned_indices_neo4j extract --property imdbVotes --output-dir data
python -m learned_indices_neo4j create-indexes --property year
python -m learned_indices_neo4j create-indexes --property imdbVotes
```

Run an exact lookup against the baseline:

```bash
python -m learned_indices_neo4j query data/year.csv --exact 1995
python -m learned_indices_neo4j query data/year.csv --exact 1995 --limit 20
python -m learned_indices_neo4j query data/year.csv --exact 1995 --index btree --limit 20
python -m learned_indices_neo4j query data/year.csv --exact 1995 --index rmi --limit 20
```

Run a range lookup:

```bash
python -m learned_indices_neo4j query data/imdbVotes.csv --min 10000 --max 100000
python -m learned_indices_neo4j query data/imdbVotes.csv --min 10000 --max 100000 --index btree
python -m learned_indices_neo4j query data/imdbVotes.csv --min 10000 --max 100000 --index rmi
```

Tune the RMI:

```bash
python -m learned_indices_neo4j tune-rmi data/year.csv --k-candidates 4,8,16,32
python -m learned_indices_neo4j tune-rmi data/imdbVotes.csv --k-candidates 4,8,16,32
python -m learned_indices_neo4j query data/year.csv --exact 1995 --index rmi --tune-rmi --limit 20
```

Show CLI help:

```bash
python -m learned_indices_neo4j
```

## Test

```bash
python -m unittest discover -s tests
```
