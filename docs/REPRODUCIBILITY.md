# Reproducibility and visual sources

## Scope

This guide describes what is available in the committed project and what is required to reproduce it. The documentation update inspected source and saved outputs; it did not rerun model training or hardware/simulation experiments.

## Required inputs

The app expects external data under `data/raw/` and `data/raw_unseen/`, result tables under `results/`, and the LoRA adapter at `chronos_lora/chronos_lora/`. The optimizer also references city metadata, forecast outputs, and the UCI household power-consumption file. These artifacts are not present in the root tree and must be recreated or supplied.

## Entry point

Use a Python environment appropriate for the PyTorch, Transformers, PEFT, Chronos, and Gradio versions used by the notebooks. A tested dependency lockfile is not committed.

Start by opening the acquisition and preprocessing notebooks in Jupyter, then follow the numbered experiments. Review their installation cells and replace local paths before execution.

After restoring the data, dependencies, and model artifacts:

```bash
python app_enhanced.py
```

The app is configured for port **7861**. It binds to `0.0.0.0`; use a local environment or review the host configuration before exposing it.

## Evidence and interpretation

The included figures are committed notebook outputs from the household-load preparation and a Jaipur energy-flow simulation. The dispatch in notebook 16 is a greedy strategy with a three-hour look-ahead heuristic; it does not establish a globally optimal schedule. Financial and CO2 KPIs depend on the notebook's tariff and emission-factor assumptions, and models were not retrained for this update.

## Visual provenance

All screenshots and experiment figures are tied to repository material. Workflow SVGs are explanatory illustrations. The full image inventory is in [assets/sources.json](assets/sources.json).

| Visual | Source |
| --- | --- |
| [overview.svg](assets/overview.svg) | New explanatory project-workflow diagram based on the committed source and documentation; not a screenshot or measured result. |
| [notebook-1.png](assets/notebook-1.png) | 16_solar_battery_optimizer.ipynb, cell 8 (1-based), committed image/png output; not rerun |
| [notebook-2.png](assets/notebook-2.png) | 16_solar_battery_optimizer.ipynb, cell 13 (1-based), committed image/png output; not rerun |

## Validation

```bash
python tools/check_docs.py
```

The standard-library checker validates README/docs local link destinations and SVG/PNG/JPEG headers. It makes no external requests and installs no project runtime dependencies. External web links, model accuracy, image semantics, and full application execution need separate review.

## Refresh notebook visuals

The exporter reads saved `image/png` outputs without executing notebook code:

```bash
python tools/export_notebook_images.py "16_solar_battery_optimizer.ipynb"
```

Review the exported images, remove any unsuitable or private content, choose the figures that explain the project, and update the README links and `docs/assets/sources.json`. It does not manufacture results when a notebook has no saved PNG output.

## Reproduction record

For a new experiment or demo, record the source commit, environment versions, data provenance and split, configuration, command/notebook order, and resulting artifacts. Only label a measurement as reproduced after completing that run. Keep credentials and restricted datasets outside version control.
