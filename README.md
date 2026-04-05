# PIFC-Diff

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

PIFC-Diff: A Physics-Informed and Feature-Conditioned Latent Diffusion Model for Self-Supervised GPR B-scan Enhancement

## Repository Structure

* `train.py`: Core model training (U-Net + Physics-Guided Loss).
* `enhance.py`: Inference script for GPR B-scan enhancement.
* `field.py`: Zero-shot application on real-world GPR data.
* `ablation.py` & `baseline.py`: Scripts to reproduce paper experiments.
* `rtm.py`: Reverse Time Migration (RTM) assessment.

## Data Availability

* **Synthetic Data:** The synthetic GPR datasets generated and used for training in this study are available upon reasonable request for academic purposes. For access, please contact the corresponding author at **xxxx@xxx.com**.
* **Field Data:** The real-world bridge deck GPR dataset used for field validation is publicly open-source. We gratefully acknowledge the 4TU.ResearchData repository for providing this dataset, which can be accessed via DOI: [10.4121/96303227-5886-41c9-8607-70fdd2cfe7c1.v1](https://doi.org/10.4121/96303227-5886-41c9-8607-70fdd2cfe7c1.v1).

## Citation

If you find this repository useful in your research, please consider citing our work:

```bibtex
@article{pifcdiff2026,
  title={PIFC-Diff: A Physics-Informed and Feature-Conditioned Latent Diffusion Model for Self-Supervised GPR B-scan Enhancement},
  author={Lastname, Firstname and Lastname, Firstname},
  journal={Name of the Journal},
  year={2026},
  volume={xx},
  number={xx},
  pages={xx--xx},
  publisher={Publisher Name},
  doi={10.xxxx/xxxxxxx}
}
