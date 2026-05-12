# CLI reference

DepthCast installs a `depthcast` console script (and is also runnable as
`python -m depthcast`). Every command supports `--help`.

```
depthcast [COMMAND] [OPTIONS]
```

## `init-config`

Write a default configuration file you can edit.

```bash
depthcast init-config [PATH]
```

| Argument | Default | Description |
|---|---|---|
| `PATH` | `depthcast.yaml` | Output path for the YAML config. |

## `synth`

Generate a synthetic FI-2010-shaped session for experimentation.

```bash
depthcast synth --out session.txt --events 20000 --seed 0
```

| Option | Default | Description |
|---|---|---|
| `--out` | `synthetic_session.txt` | Destination text file. |
| `--events` | 20000 | Number of order-book events. |
| `--seed` | 0 | Random seed. |

## `summary`

Print the architecture and parameter counts for a configuration.

```bash
depthcast summary --config my_run.yaml
```

| Option | Default | Description |
|---|---|---|
| `--config` | built-in defaults | Config YAML to build the model from. |

## `train`

Train a model and checkpoint the best validation epoch.

```bash
# On the FI-2010 training file, holding out 20% for validation
depthcast train --config my_run.yaml \
  --train-file Train_Dst_NoAuction_DecPre_CF_7.txt --val-fraction 0.2

# On synthetic data, no dataset required
depthcast train --config my_run.yaml --synthetic --synthetic-events 8000
```

| Option | Default | Description |
|---|---|---|
| `--config` | built-in defaults | Config YAML. |
| `--train-file` | none | FI-2010 training matrix. |
| `--val-file` | none | Optional separate validation matrix. |
| `--val-fraction` | 0.2 | Holdout fraction when no `--val-file`. |
| `--synthetic` | false | Train on generated data. |
| `--synthetic-events` | 20000 | Events to synthesise. |

Provide either `--train-file` or `--synthetic`.

## `evaluate`

Score a checkpoint and print a classification report.

```bash
depthcast evaluate --checkpoint checkpoints/best.pt \
  --test-file Test_Dst_NoAuction_DecPre_CF_7.txt
```

| Option | Default | Description |
|---|---|---|
| `--checkpoint` | required | Checkpoint file to evaluate. |
| `--test-file` | none | FI-2010 test matrix. |
| `--synthetic` | false | Evaluate on generated data. |
| `--synthetic-events` | 10000 | Events to synthesise. |

Provide either `--test-file` or `--synthetic`.

## Shell completion

Typer-based completion is available for bash, zsh, and fish:

```bash
depthcast --install-completion
```

## Exit codes and logging

Commands exit non-zero on invalid arguments or missing files, with a
descriptive message. Progress is logged to standard error through the
structured logger; final results (parameter counts, best epoch, metric
report) are written to standard output.
