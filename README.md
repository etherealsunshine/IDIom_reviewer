# IDiom RL Audit

Reproducible audit scripts for testing whether IDiom post-training with the
ProtGPS reward learns compartment-specific IDR grammar or shallow reward-model
shortcuts.

The code focuses on:

- composition-matched full and block shuffles
- shallow-feature RandomForest probes of ProtGPS scores
- motif- and composition-matched cheap baselines
- ProtGPS off-target/specificity analysis
- DeepLoc 2.x independent validation

See [README_AUDIT.md](README_AUDIT.md) for the full workflow notes and commands.

## Quick Start

```bash
uv sync
UV_CACHE_DIR=.uv-cache uv run idiom-audit --help
```

The external IDiom repository is tracked as a submodule:

```bash
git submodule update --init --recursive
```

Large downloaded assets, generated CSVs, plots, logs, model checkpoints, and
third-party software tarballs are intentionally ignored. Modal workflows cache
large model/data assets in Modal Volumes instead of committing them to Git.

## Repository Layout

```text
idiom_audit/     importable analysis package
tools/modal/     Modal GPU scoring and external-predictor jobs
tools/protgps/   local ProtGPS scoring wrapper
tools/workflows/ shell workflows for unattended runs
data/            lightweight download notes; generated data is ignored
results/         generated analysis outputs, ignored by Git
```
