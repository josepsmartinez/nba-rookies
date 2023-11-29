#!/bin/bash

# Code tested with Python 3.11 on Ubuntu 20.04

.venv/bin/python -m mypy ifrecog
.venv/bin/python -m flake8 ifrecog
