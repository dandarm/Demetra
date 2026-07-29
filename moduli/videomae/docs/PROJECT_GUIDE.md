# DeMeTrA Project Guide

This guide explains how the main components of DeMeTrA fit together. The
[project README](../README.md) provides the concise overview, quick-start
commands, publication details, and the complete documentation index. The
individual pages in this directory contain the operational details for each
workflow.

## 1. Project scope

DeMeTrA is a self-supervised vision-transformer framework for detecting and
tracking Mediterranean tropical-like cyclones (Medicanes) in SEVIRI/MSG
AirmassRGB satellite imagery. It adapts VideoMAE v2 to a domain-specific
pipeline with three main learning stages:

1. self-supervised specialization on unlabelled satellite sequences;
2. supervised fine-tuning for cyclone detection;
3. coordinate regression for cyclone-centre tracking.

The project also includes the data-building, distributed execution, inference,
evaluation, and visual-quality-assurance tools needed to support these stages.

## 2. End-to-end workflow

```text
SEVIRI/MSG imagery + MANOS tracks
                |
                v
      dataset preparation
                |
                v
 self-supervised specialization
                |
                v
   detection fine-tuning
                |
                v
 cyclone-centre tracking
                |
                v
 inference, metrics, and visual QA
```

The stages can be used independently. For example, an existing checkpoint can
be taken directly to folder-based inference, while a new dataset generally
requires the complete path from geospatial preparation through training and
evaluation.

## 3. Repository architecture

The main command-line entry points are:

* [`specialization.py`](../specialization.py) for self-supervised domain
  specialization;
* [`classification.py`](../classification.py) for supervised detection;
* [`tracking.py`](../tracking.py) for cyclone-centre regression;
* [`inference_classification.py`](../inference_classification.py) for
  classification inference;
* [`predict_and_track_from_folder.py`](../predict_and_track_from_folder.py) for
  the combined operational detection-and-tracking workflow.

Supporting code is organised by responsibility:

* [`dataset/`](../dataset/) contains dataset construction, loading, sampling,
  and augmentation;
* [`models/`](../models/) contains the VideoMAE-based model definitions;
* [`medicane_utils/`](../medicane_utils/) contains satellite, geographic, and
  Medicane-specific utilities;
* the `engine_for_*.py` modules implement training loops;
* [`model_analysis.py`](../model_analysis.py), [`plot_results.py`](../plot_results.py),
  and the analysis notebooks cover evaluation and visualisation.

The [training call tree](training_call_tree.md) and its associated diagrams
provide a more detailed view of script dependencies.

## 4. Data preparation

Dataset preparation connects satellite imagery to the temporal and geographic
annotations used by the learning tasks.

### Track preparation

The [MANOS track analysis](Analyze_Manos_tracks.md) describes how track files
are aggregated, assigned consistent identifiers, mapped from geographic
coordinates to image pixels, and adjusted to suitable Medicane time windows.
The [pixel-to-kilometre conversion guide](pixel_km_conversion.md) and
[geographic utilities reference](medicane_utils_geo_const.md) document the
coordinate transformations used throughout the project.

### Video dataset construction

The [dataset-building guide](Build_dataset_videoMAE.md) covers tile extraction,
16-frame sequence creation, balancing, and storage for supervised and
self-supervised use. The [AirmassRGB dataset command](make_dataset_from_rgb.md)
acts as the dispatcher for master dataframes, video tiles, cloudy subsets,
annual datasets, and MANOS-derived datasets.

Additional workflows support:

* cloud-cover estimation and annotation in [Cloud index](Cloud_index.md);
* relabelling and custom splits in [Dataset experiments](Experiment_dataset.md);
* manual cyclone-window updates in [Cyclone video cuts](Video_cyclones_cut.md).

## 5. Training

### Self-supervised specialization

The [pretraining reference](PRETRAIN.md) describes distributed VideoMAE
pretraining and its principal masking and optimisation parameters. The
[specialization workflow](specialization.md) explains how `specialization.py`
loads a pretrained MAE checkpoint, constructs the satellite dataset, and runs
domain-specific training and validation.

### Cyclone detection

The [classification workflow](classification.md) follows the supervised path
from command-line arguments through dataloaders, the fine-tuning engine,
validation, and checkpointing.

### Cyclone-centre tracking

The [tracking task reference](TRACKING.md) describes the regression dataset,
model, targets, and loss. The [tracking workflow](tracking.md) documents the
training entry point, distributed execution, validation, and checkpoints.

For large runs, see the [distributed training guide](distributed_training_summary.md).
The [learning-rate overview](learning_rate_overview.md) explains how schedules
and effective learning rates are derived.

## 6. Inference and post-processing

The repository provides both task-specific and end-to-end inference:

* [classification inference](inference_classification.md) produces predictions,
  logits, or embeddings and supports distributed result aggregation;
* [inference from a folder](inference_from_folder.md) documents the general
  folder-based interface;
* [detection and tracking from a folder](predict_and_track_from_folder.md)
  connects the two models in a single operational workflow;
* [tracking from a folder](tracking_from_folder.md) focuses on regression-only
  execution;
* [general prediction analysis](Predict_general_data.md) covers custom
  inference datasets and downstream processing.

The resulting predictions can be joined back to the source frames, projected
over the Mediterranean domain, and exported as images, GIFs, or videos.

## 7. Evaluation and visual quality assurance

DeMeTrA uses both numerical and visual evaluation:

* [performance metrics](metrics.md) explains POD/Recall, FAR, CSI, HSS, and
  balanced accuracy for rare-event detection;
* [model statistics](Model_stats.md) covers validation-set construction and
  detailed error analysis;
* [training-loss plots](Plot_train_loss.md) and
  [metric comparisons](Plot_compare_metrics.md) compare experiments and
  learning dynamics;
* [Mediterranean validation predictions](View_MED_val_preds.md),
  [Mediterranean tracking predictions](View_MED_tracking_preds.md),
  [test-tile visualisation](View_test_tiles.md), and
  [tracking-tile visualisation](View_tracking_tiles.md) support geographic and
  temporal QA;
* [patch verification](Verifica_patches.md) inspects MAE reconstructions,
  masks, and specialised-pretraining behaviour.

The repository root also contains
[supplementary material](../SUPPLEMENTARY_MATERIAL.md) focused on Ianos and
MANOS track–image consistency.

## 8. Recommended reading paths

### Reproduce the complete project

1. Follow the [installation guide](INSTALL.md).
2. Prepare tracks and satellite data using the data-building documentation.
3. Run specialization, classification, and tracking in that order.
4. Execute inference on held-out data.
5. Compare checkpoints using the metric and visual-analysis workflows.

### Apply trained models to new imagery

1. Confirm that the imagery follows the expected AirmassRGB format and
   geospatial conventions.
2. Build the required clips or use folder-based inference.
3. Run detection and tracking.
4. Inspect both numerical metrics and spatial overlays.

### Extend one component

Start from the relevant entry-point guide, then use the call tree to identify
the training engine, dataset class, and model implementation affected by the
change. This keeps modifications scoped while preserving the rest of the
pipeline.

## 9. Publication and upstream work

The methods and experiments are described in the
[DeMeTrA preprint](https://www.preprints.org/manuscript/202605.1494). Citation
metadata is available in [`CITATION.cff`](../CITATION.cff).

DeMeTrA builds on
[VideoMAE v2](https://github.com/OpenGVLab/VideoMAEv2). The upstream papers
remain relevant when describing the underlying masked-autoencoder architecture;
the DeMeTrA preprint is the primary reference for this repository's
domain-specific pipeline and results.
