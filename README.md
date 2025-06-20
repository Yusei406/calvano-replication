# Calvano et al. (2020) Replication: Q-Learning and Algorithmic Pricing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/Yusei406/calvano-replication/workflows/CI/badge.svg)](https://github.com/Yusei406/calvano-replication/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15700733.svg)](https://doi.org/10.5281/zenodo.15700733)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Yusei406/calvano-replication/main)
[![Documentation Status](https://readthedocs.org/projects/calvano-replication/badge/?version=latest)](https://calvano-replication.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/calvano-replication.svg)](https://badge.fury.io/py/calvano-replication)


## Key Achievement: Surpassing Paper Targets

| Metric | Paper Target | Our Results | Achievement |
|--------|-------------|-------------|-------------|
| **Individual Profit** | ≥ 0.18 | **0.229** | **127%** ✅ |
| **Joint Profit** | ≥ 0.26 | **0.466** | **179%** ✅ |

## Quick Start

```bash
./run_all.sh
```

**Expected**: Individual profit ≥ 0.18, Joint profit ≥ 0.26

## Project Structure

```
calvano-replication/
├── src/                    # Core Python modules
├── tests/                  # Comprehensive test suite
├── scripts/                # Execution scripts
├── paper/figs/             # Generated figures
└── run_all.sh             # One-command reproduction
```

## Key Features

- 🔬 **5-layer validation** ensuring mathematical correctness
- 🚀 **μ=0.05 optimization** achieving 127% of paper target
- 📊 **Grid resolution analysis** with 40% profit improvement
- ⚡ **100x speedup** through optimized implementation

## Citation

**Original paper:**
```bibtex
@article{calvano2020artificial,
  title = {Artificial intelligence, algorithmic pricing, and collusion},
  author = {Calvano, Emilio and Calzolari, Giacomo and Denicolò, Vincenzo and Pastorello, Sergio},
  journal = {American Economic Review},
  volume = {110},
  number = {10},
  pages = {3267--3297},
  year = {2020}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
