# LDC Demand Forecasting :chart_with_upwards_trend:

Forecasting gas demand distributed via local gas networks in the UK.

All analysis and results are contained in the notebook `main.ipynb`. The source directory `src/` contains some helper code, including the Bayesian linear regression model, linear basis mapping and some evaluation metrics. The provided datasets can be found in the `data/` directory.

A good reference on BLR can be found in chapter 9 of [Maths for Machine Learning](https://mml-book.github.io/book/mml-book.pdf).

Apologies for the lack of docstrings and proper commenting / documentation, I was quite pressed for time!

## Installation :computer: 

To install the environment, you will need to have python 3.10 installed as well as [`poetry`](https://python-poetry.org) installation. The `tensorflow` dependencies are for MacOS -- simply replace `tensorflow-macos` with `tensorflow` in the `pyproject.toml` file for installation on Linux or Windows. From the top-level directory, run the following commands:

```bash
poetry env use <path to python 3.10 executable>
poetry install
```

## Launching the Jupyter Notebook :rocket:

With the environment activated, run:

```bash
jupyter notebook
```
