# Black Friday MLOps Project

This repository contains an end-to-end MLOps pipeline for Black Friday sales prediction. It includes data ingestion, preprocessing, model training, deployment, monitoring, and CI/CD integration.

## Project Structure
- `src/` - Source code for data, models, API, and utilities
- `data/` - Raw, processed, and feature data
- `models/` - Model experiments and registry
- `deployment/` - Docker and AWS deployment files
- `monitoring/` - Dashboards and alerts
- `tests/` - Unit and integration tests
- `docs/` - Documentation
- `notebooks/` - Jupyter notebooks
- `infrastructure/` - Infrastructure as code
- `scripts/` - Helper scripts

## Setup
1. Clone the repository
2. Create and activate a Python virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Configure AWS CLI: `aws configure`

## Usage
- Train models, serve predictions, and monitor performance using provided scripts and notebooks.

## AWS CLI Access
To use AWS CLI, activate your virtual environment and run commands like:
```
aws s3 ls
```
Refer to the documentation in `docs/` for more details.
