# learned indices neo4j

A Python project for experimenting with learned indexing ideas on graph data in Neo4j.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Copy `.env.example` to `.env` and update the Neo4j connection values.

Index and tuning hyperparameters live in `experiment.toml`.

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

Default hyperparameters:

```toml
[btree]
order = 64

[rmi]
k = 16
delta = 0
auto_delta = true

[rmi.tuning]
k_candidates = [5, 10, 20, 50]
delta_candidates = [10, 25, 50, 100]
folds = 5

[experiment]
seed = 42
train_fraction = 0.8
query_count = 200
workload_dir = "workloads"
results_dir = "results"
```

CLI flags override `experiment.toml` for one-off runs.

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

Generate the 200-query point lookup workloads from the held-out 20% test split:

```bash
python -m learned_indices_neo4j generate-workload
```

This writes:

- `workloads/year_point_queries.csv`
- `workloads/imdbVotes_point_queries.csv`

Run the full experiment and generate report tables:

```bash
python -m learned_indices_neo4j run-experiment
```

This writes:

- `results/main_comparison.csv`
- `results/rmi_hyperparameter_sensitivity.csv`
- `results/rmi_worst_case_queries.csv`
- `results/lookup_latency_detail.csv`

`main_comparison.csv` includes index build time, total evaluation execution time, average/min/max
per-lookup latency, average elements examined, RMI MAE, RMI coverage, and the selected RMI
hyperparameters.

`lookup_latency_detail.csv` has one row per sampled lookup per index. Use it to compare whether
RMI fetched/found individual point lookups faster than the B+ tree. It includes the property, query
value, index type, elements examined, found/covered flags, and measured `lookup_latency_ms`.

The experiment runner uses an 80/20 train/test split. The RMI is trained on the 80% training split,
point queries are sampled from the 20% test split, and RMI hyperparameters are cross-validated on
the training split. The grid defaults are `k = [5, 10, 20, 50]` and
`delta = [10, 25, 50, 100]`.

If no configuration in the configured `delta` grid covers every validation query, the runner
chooses the highest-coverage configuration and reports `RMI coverage` in the main table. Increase
`delta_candidates` if you want correctness-constrained runs with full coverage.

Use a different config file:

```bash
python -m learned_indices_neo4j --config experiment.toml query data/year.csv --exact 1995 --index rmi
```

Show CLI help:

```bash
python -m learned_indices_neo4j
```

## Test

```bash
python -m unittest discover -s tests
```
