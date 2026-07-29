# SCoPe: ZTF Source Classification Project

[![PyPI version](https://badge.fury.io/py/scope-ml.svg)](https://badge.fury.io/py/scope-ml)
[![arXiv](https://img.shields.io/badge/arXiv-2102.11304-blue)](https://arxiv.org/abs/2102.11304)
[![arXiv](https://img.shields.io/badge/arXiv-2009.14071-blue)](https://arxiv.org/abs/2009.14071)
[![arXiv](https://img.shields.io/badge/arXiv-2312.00143-blue)](https://arxiv.org/abs/2312.00143)

`scope-ml` uses machine learning to classify variable star light curves from the Zwicky Transient Facility ([ZTF](https://www.ztf.caltech.edu)) and the Vera C. Rubin Observatory ([LSST](https://rubinobs.org)).

## What SCoPe does

- **Feature generation** from light curves: period-finding (Conditional Entropy, Analysis of Variance, Lomb-Scargle, FPW), Fourier decomposition, and statistical features via the [periodfind](https://github.com/ZwickyTransientFacility/periodfind) library
- **Classification** using XGBoost and Deep Neural Network binary classifiers trained on ~80,000 manually-labeled sources
- **Integration** with the [Fritz](https://fritz.science) transient broker for uploading/downloading classifications
- **Scalable processing** via SLURM for large-scale feature generation and inference across ZTF fields

## Supported data sources

- **ZTF** light curves via the [Kowalski](https://github.com/skyportal/kowalski) database
- **Rubin DP1** forced photometry via TAP API or local parquet files
- **External catalogs**: Gaia EDR3, AllWISE, Pan-STARRS1

## Quick links

- [Installation](getting-started/installation.md) -- get started with `pip install scope-ml`
- [Quick Start](getting-started/quickstart.md) -- train your first classifier in minutes
- [Feature Generation](user-guide/feature-generation.md) -- generate features from ZTF light curves
- [Rubin DP1](user-guide/rubin-dp1.md) -- process Rubin Data Preview 1 data
- [Rubin EDP2/DP2](user-guide/rubin-dp2.md) -- process Rubin Early Data Preview 2 DIA photometry
- [Field Guide](field-guide/index.md) -- learn about the source classes SCoPe identifies
- [CLI Reference](reference/cli.md) -- all available commands

## Funding

We gratefully acknowledge previous and current support from the U.S. National Science Foundation (NSF) Harnessing the Data Revolution (HDR) Institute for [Accelerated AI Algorithms for Data-Driven Discovery (A3D3)](https://a3d3.ai) under Cooperative Agreement No. [PHY-2117997](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2117997).
