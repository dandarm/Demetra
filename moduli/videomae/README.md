<div align="center">
<h1>DeMeTrA: Detection and Tracking of Medicanes</h1>
<br>
<img src="misc/med_earth_icon.png" alt="Project Icon" width="100" />
</div>

# Overview

**DeMeTrA** is a self-supervised vision-transformer framework for detecting and tracking *Mediterranean tropical-like cyclones* (Medicanes) from SEVIRI/MSG AirmassRGB satellite imagery. It provides workflows for self-supervised pretraining, supervised cyclone detection, and cyclone-centre tracking.

> **Preprint:** [*Detection and Tracking of Medicanes Through DeMeTrA Self-Supervised Vision Transformer*](https://www.preprints.org/manuscript/202605.1494) — Daniele D’Armiento, Stefano Sebastianelli, Leo Pio D’Adderio, Paolo Sanò, Daniele Casella, and Giulia Panegrossi (2026). DOI: [10.20944/preprints202605.1494.v1](https://doi.org/10.20944/preprints202605.1494.v1)

The repo is developed to:

* Pretrain the model for a **specialization phase** using unlabeled satellite video sequences.
* Fine-tune the pretrained model for **cyclone detection** and **center tracking**.

---

## Repository Structure

```
├── specialization.py            # Additional unsupervised pretraining ("specialization")
├── classification.py            # Fine-tuning entry point (classification)
├── tracking.py                  # Cyclone-centre regression training
├── inference_classification.py  # Classification inference
├── predict_and_track_from_folder.py # End-to-end operational inference
├── engine_for_pretraining.py    # Training loop for pretraining
├── engine_for_finetuning.py     # Training loop for fine-tuning
├── engine_for_tracking.py       # Training loop for tracking
├── dataset/                     # Dataset building, loading, and augmentation
├── models/                      # VideoMAE-based model definitions
├── medicane_utils/              # Satellite and geospatial utilities
├── make_dataset_from_rgb.py     # Entry points for dataset construction
├── docs/                        # Project and workflow documentation
└── arguments.py                 # Centralised training arguments
```

The [project guide](docs/PROJECT_GUIDE.md) explains how these components fit
together across the complete data, training, inference, and evaluation
pipeline.

---

## Documentation

The following index collects the project documentation, notebooks, and operational notes.

### Project overview and setup

* [Extended project guide](docs/PROJECT_GUIDE.md)
* [Installation](docs/INSTALL.md)
* [Supplementary material: Ianos and MANOS track–image consistency](SUPPLEMENTARY_MATERIAL.md)

### Training workflows

* [Self-supervised pretraining](docs/PRETRAIN.md)
* [Specialization workflow](docs/specialization.md)
* [Classification training](docs/classification.md)
* [Cyclone-centre tracking](docs/tracking.md)
* [Distributed multi-GPU and multi-node training](docs/distributed_training_summary.md)
* [Training call tree](docs/training_call_tree.md)
* [Learning-rate mechanics](docs/learning_rate_overview.md)

### Dataset construction and preparation

* [Building the VideoMAE dataset](docs/Build_dataset_videoMAE.md)
* [Dataset experiments and relabelling](docs/Experiment_dataset.md)
* [Building datasets from AirmassRGB imagery](docs/make_dataset_from_rgb.md)
* [Cloud-index analysis](docs/Cloud_index.md)
* [Geographic constants and utilities](docs/medicane_utils_geo_const.md)
* [Pixel-to-kilometre coordinate conversion](docs/pixel_km_conversion.md)

### Inference and operational workflows

* [Classification inference](docs/inference_classification.md)
* [Inference from a folder](docs/inference_from_folder.md)
* [Detection and tracking from a folder](docs/predict_and_track_from_folder.md)
* [Tracking from a folder](docs/tracking_from_folder.md)
* [General prediction workflow](docs/Predict_general_data.md)

### Analysis, evaluation, and visualisation

* [MANOS track analysis](docs/Analyze_Manos_tracks.md)
* [Performance metrics](docs/metrics.md)
* [Model statistics](docs/Model_stats.md)
* [Training-loss visualisation](docs/Plot_train_loss.md)
* [Metrics comparison plots](docs/Plot_compare_metrics.md)
* [Mediterranean tracking predictions](docs/View_MED_tracking_preds.md)
* [Mediterranean validation predictions](docs/View_MED_val_preds.md)
* [Test-tile visualisation](docs/View_test_tiles.md)
* [Tracking-tile visualisation](docs/View_tracking_tiles.md)
* [Patch verification](docs/Verifica_patches.md)
* [Cyclone-video generation](docs/Video_cyclones_cut.md)

### Project notes

* [Completed work and metrics](misc/done.md)
* [Open tasks](misc/todo.md)

---

## Datasets

* **Unsupervised pretraining**: uses unlabeled video sequences (AirmassRGB), with loss for patch reconstruction
* **Supervised fine-tuning**: uses Medicanes tracks file for 
    * **classification training** for detection,
    * **Regression training** for center tracking.

---

# How to Run

## Installation

Please follow the instructions in [INSTALL.md](docs/INSTALL.md).

### Pretraining
```bash
python specialization.py [OPTIONS...]
```

### Fine-tuning: Medicane Detection 
```bash
python classification.py [OPTIONS...]
```

### Fine-tuning: Center Tracking 
```bash
python tracking.py [OPTIONS...]
```

### Inference From Folder
See [`docs/inference_from_folder.md`](docs/inference_from_folder.md).

---




## Download and Processing of AirmassRGB

To download and process **EUMETSAT satellite images** into **AirmassRGB composites**, use the script:


```bash
python medicane_utils/download_airmassRGB.py --start "2020-09-01 00:00" --end "2020-09-15 23:59"
```


## Citation

If you use DeMeTrA, please cite the preprint:

> D’Armiento, D., Sebastianelli, S., D’Adderio, L. P., Sanò, P., Casella, D., & Panegrossi, G. (2026). *Detection and Tracking of Medicanes Through DeMeTrA Self-Supervised Vision Transformer*. Preprints.org. https://doi.org/10.20944/preprints202605.1494.v1

GitHub also exposes this citation through [`CITATION.cff`](CITATION.cff).

## Upstream work

DeMeTrA builds on [VideoMAE v2](https://github.com/OpenGVLab/VideoMAEv2). Please also cite the upstream work where appropriate:

* [VideoMAE v1 (NeurIPS 2022)](https://arxiv.org/abs/2203.12602)
* [VideoMAE v2 (CVPR 2023)](https://arxiv.org/abs/2303.16727)
