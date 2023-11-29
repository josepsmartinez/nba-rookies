# NBA's 2022-2023 season rookies facial recognition

This repository employs facial biometry models from InsightFace to perform facial recognition on NBA's 2023 best performing rookies, Victor Wembanyama and Scoot Henderson. Target subjects are compared among themselves as well as NBA star Ja Morant.

## Setup

To respectively download models and create a Python virtual environment, run the scripts `download_models.sh` and `install_requirements.sh`.

## Running demos

Be sure to include the repository's root in your Python interpreter's path list.
For example, to run the face verification demo from the repository's root, execute:

```bash
PYTHONPATH=. .venv/bin/python demos/verification.py
```

## Validating code quality

One can further statically analyze code for linting (`flake8`) and type checking (`mypy`) by installing libraries at `requirements/dev.txt` and running the script `static_check.sh`.
