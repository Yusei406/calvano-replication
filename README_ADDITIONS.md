
## 🐳 Docker Quick Start

### One-Command Reproduction
```bash
git clone https://github.com/Yusei406/calvano-replication.git
cd calvano-replication

# Full reproduction in isolated container
docker-compose run --rm calvano-replication

# Quick test (5 minutes)  
docker-compose run --rm quick-test

# Interactive Jupyter environment
docker-compose up jupyter  # Access at http://localhost:8888
```

## 🐍 Conda Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate calvano-replication

# Verify installation
python -c "import src.environment; print('✅ Conda ready')"
```

## 🔧 Troubleshooting & FAQ

### Common Issues

#### Import Errors
```bash
# Error: "ModuleNotFoundError: No module named 'src'"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or run with explicit path
python -m sys.path.append('src')
```

#### macOS Apple Silicon (M1/M2)
```bash
# If NumPy/Numba fails on ARM64
conda install numpy numba -c conda-forge

# Alternative: Use Rosetta mode
arch -x86_64 python -m pip install -r requirements.txt
```

#### Windows WSL
```bash
# Enable WSL2 and install Ubuntu
wsl --install -d Ubuntu

# Then follow Linux instructions
```

#### Memory Issues
```bash
# Reduce episodes for testing
python scripts/sweep_mu_precision.py --episodes 1000 --runs 1

# Monitor memory usage
htop  # Linux/macOS
taskmgr  # Windows
```

#### Performance Optimization
```bash
# Enable parallel processing
export OMP_NUM_THREADS=4

# Use Numba JIT (automatic in our implementation)
python -c "import numba; print(f'Numba version: {numba.__version__}')"
```

### Expected Results Verification

#### Minimum Acceptance Criteria
```python
# Expected results (μ=0.05 configuration)
assert individual_profit >= 0.18  # Paper target
assert joint_profit >= 0.26       # Paper target  
assert convergence_rate >= 0.90    # Convergence threshold

# Our typical results
individual_profit ≈ 0.229  # 127% of target
joint_profit ≈ 0.466       # 179% of target
convergence_rate = 1.0      # 100% convergence
```

#### Debugging Failed Reproduction
```bash
# Check system info
python -c "
import sys, platform, numpy
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')  
print(f'NumPy: {numpy.__version__}')
"

# Run minimal test
python -c "
import sys; sys.path.append('src')
from environment import *
from qlearning import *
print('✅ Core imports successful')
"

# Validate numerical precision
python tests/test_symbolic_vs_numeric.py
```

### Performance Benchmarks

| Environment | Episodes/sec | Memory (MB) | Setup Time |
|-------------|-------------|-------------|------------|
| **Local Python** | 5,000 | 100 | 2 min |
| **Docker** | 4,500 | 150 | 5 min |
| **Conda** | 5,200 | 120 | 3 min |
| **HPC/Cluster** | 8,000+ | 80 | 1 min |

### Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/Yusei406/calvano-replication/issues/new/choose)
- **Discussions**: [Q&A and feature requests](https://github.com/Yusei406/calvano-replication/discussions)
- **Academic Support**: For research collaborations, cite our DOI: `10.5281/zenodo.15700733`

---

## 🏷️ Repository Topics

This repository is tagged with the following topics for discoverability:

`q-learning` `algorithmic-pricing` `machine-learning` `economics` `replication` `multi-agent-systems` `pricing-algorithms` `collusion` `reinforcement-learning` `duopoly` `artificial-intelligence` `computational-economics` `academic-research` `python` `numba` `simulation` `game-theory` `market-dynamics`

### Academic Keywords
- **Economics**: Algorithmic pricing, collusion, duopoly, market competition
- **Computer Science**: Q-learning, multi-agent systems, reinforcement learning
- **Methodology**: Replication study, computational economics, simulation
- **Technology**: Python, Numba JIT, Docker, scientific computing

