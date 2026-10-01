# projeto_kunumi

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         meta_curation and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── meta_curation   <- Source code for use in this project.
    │
    ├── __init__.py             <- Re-exports public DataRater API
    │
    ├── types.py                <- Shared aliases (ParamDict, Batch, ...)
    ├── config.py               <- DataRaterConfig (§3 of single-file.py)
    ├── optim.py                <- Differentiable inner optimisers (§1)
    ├── meta_optim.py           <- MetaAdam meta-optimiser (§2)
    ├── trainer.py              <- DataRaterTrainer, Algorithm 1 (§4)
    ├── filtering.py            <- Top-K / CDF filtering utilities (§5)
    ├── demo.py                 <- Toy demo (§7, `python -m meta_curation.demo`)
    ├── single-file.py          <- Reference single-file implementation (kept as reference)
    │
    └── modeling
        ├── __init__.py
        └── models.py           <- Example DataRater nets (§6)
```

--------

