# AGENTS.md

## Purpose
This file provides a **navigation map for agents and automation tools** working on this repository. It does not define coding conventions, testing frameworks, or rigid engineering practices. Instead, it points to the key entry points and detailed documentation resources.

For extended explanations, see:
- [README.md](README.md)
- [Project guide](docs/PROJECT_GUIDE.md)

---

## Main Entry Points
- **Specialization (self-supervised pretraining)** → [`specialization.py`](specialization.py)  
  Run: `python specialization.py [OPTIONS...]`  
  Documentation: [specialization.md](docs/specialization.md)

- **Classification fine-tuning (cyclone detection)** → [`classification.py`](classification.py)  
  Run: `python classification.py [OPTIONS...]`  
  Documentation: [classification.md](docs/classification.md)

- **Cyclone center tracking (regression)** → [`tracking.py`](tracking.py)  
  Run: `python tracking.py [OPTIONS...]`  
  Documentation: [tracking.md](docs/tracking.md)

- **Inference** → [`inference_classification.py`](inference_classification.py)  
  Run: `python inference_classification.py [OPTIONS...]`  
  Documentation: [inference_classification.md](docs/inference_classification.md)

---

## Dataset Pipeline
- **Building datasets**: [Build_dataset_videoMAE.md](docs/Build_dataset_videoMAE.md)
- **Dataset experiments / relabeling**: [Experiment_dataset.md](docs/Experiment_dataset.md)
- **Cyclone reference-track analysis**: [reference_track_analysis.md](docs/reference_track_analysis.md)

---

## Training Workflows
- **Pretraining (MAE auto-supervised)**: [specialization.md](docs/specialization.md)
- **Fine-tuning (classification)**: [classification.md](docs/classification.md)
- **Tracking (regression)**: [tracking.md](docs/tracking.md)

Cluster jobs (HPC/Slurm) are referenced in these docs.

---

## Inference and Post-processing
- [inference_classification.md](docs/inference_classification.md)
- [Predict_general_data.md](docs/Predict_general_data.md)
- [View_MED_val_preds.md](docs/View_MED_val_preds.md)
- [View_test_tiles.md](docs/View_test_tiles.md)

---

## Tracking Visualization
- [View_tracking_tiles.md](docs/View_tracking_tiles.md)

---

## Evaluation and Metrics
- [metrics.md](docs/metrics.md)
- [Model_stats.md](docs/Model_stats.md)

---

## Notes
- No testing framework, linting rules, or formatting conventions are enforced.  
- This is a **research-driven project**: flexibility and exploration take precedence over rigid structure.  
- Agents should use this file as an **index** to locate the right documentation and scripts.
