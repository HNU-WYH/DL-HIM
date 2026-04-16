## 🚩Introduction

This repository provides the official implementation of the research paper: [**"Are Deep Learning Based Hybrid PDE Solvers Reliable? Why Training Paradigms and Update Strategies Matter"**](https://arxiv.org/abs/2602.06842).

------

## 📂 Repository Structure

The project is organized as follows:

- `scripts/`: Reproducible scripts for the results and figures presented in the paper.
- `dataset/`: Data generation tools and datasets for neural operator training.
- `checkpoints/`: Pre-trained model weights for HINTS and FNS.
- `configs/`: Hyperparameters and configuration files for the DL-HIM solvers.
- `src/`: Core source code:
  - `data_generation/`: GRF-based sampling for $k(x)$ and $f(x)$.
  - `neural_operator/`: Implementations of **DeepONet-based HINTS** and **FFT-based FNS**.
  - `training/`: Static (offline) and dynamic (unrolled) training frameworks.
  - `stepin_utils/`: Implementation of Standard AA and **Physics-Aware AA (PA-AA)**.
  - `problems/`: Numerical setups for 1D Stochastic Diffusion and Indefinite Helmholtz equations.

------

## 📝 Citation

If this work helps your research into making AI-based PDE solvers more reliable, please cite:

```
@misc{wu2026deeplearningbasedhybrid,
      title={Are Deep Learning Based Hybrid PDE Solvers Reliable? Why Training Paradigms and Update Strategies Matter}, 
      author={Yuhan Wu and Jan Willem van Beek and Victorita Dolean and Alexander Heinlein},
      year={2026},
      eprint={2602.06842},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2602.06842}, 
}
```
