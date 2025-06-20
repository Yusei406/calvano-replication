# Contributing to Calvano et al. (2020) Replication

Thank you for your interest in contributing to this academic replication project!

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/calvano-replication.git
cd calvano-replication
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Code Standards

- **Type Hints**: All functions must include type annotations
- **Docstrings**: Google-style docstrings required
- **Test Coverage**: New features require ≥90% coverage
- **Formatting**: Use black, isort, ruff for code style

## Testing

```bash
pytest tests/ -v --cov=src
```

## Academic Standards

- Fixed random seeds for reproducibility
- Exact dependency versions
- Numerical precision to 6 decimal places
- Academic references for new methods

For detailed guidelines, see the full documentation in this file.
