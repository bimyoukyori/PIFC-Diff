# PIFC-Diff

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

PIFC-Diff: A Physics-Informed and Feature-Conditioned Latent Diffusion Model for Self-Supervised GPR B-scan Enhancement

## Repository Structure

* `train.py`: Core model training (U-Net + Physics-Guided Loss).
* `enhance.py`: Inference script for GPR B-scan enhancement.
* `field_apply.py`: Zero-shot application on real-world GPR data.
* `ablation.py` & `baseline.py`: Scripts to reproduce paper experiments.
* `rtm_assess.py`: Reverse Time Migration (RTM) assessment.

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
