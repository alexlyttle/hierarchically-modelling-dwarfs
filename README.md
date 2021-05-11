# Hierarchically Modelling Dwarfs

[![DOI](https://zenodo.org/badge/305650080.svg)](https://zenodo.org/badge/latestdoi/305650080) [![arXiv badge](https://img.shields.io/badge/arXiv-2105.04482-red)](https://arxiv.org/abs/2105.04482)

Using machine learning to hierarchically model the first APOKASC catalogue of *Kepler* dwarf and subgiant stars ([Serenelli et al. 2017](https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract)). A new method which encodes the population distribution of helium and mixing-length parameter in the solar neighbourhood.

- [Hierarchically Modelling Dwarfs](#hierarchically-modelling-dwarfs)
  - [About](#about)
  - [Requirements](#requirements)
  - [Installation and running the code](#installation-and-running-the-code)
  - [Contributing](#contributing)
  - [Paper](#paper)
  - [Contact](#contact)
  - [Acknowledgments](#acknowledgments)

## About

This repository is the working notebook concerning the hierarchical modelling of low-mass dwarfs in the solar neighbourhood using the machine learning of stellar models. The repository contains Jupyter notebooks and python scripts written for the project.

## Requirements

Requires Python >= 3.5 and a number of packages found in `hierarchically-modelling-dwarfs/requirements.txt`. An unpublished package called `interstellar` is also used for the work in `hierarchically-modelling-dwarfs/training` and `hierarchically-modelling-dwarfs/modelling`. Whilst this package is not publicly available, please contact me for details if you wish to run this code. We aim to publish the package in the future if it would be useful to the wider community.

## Installation and running the code

To download the repository, run the following command in a terminal,

```terminal
git clone https://github.com/alexlyttle/hierarchically-modelling-dwarfs.git
```

Much of the work is written in Jupyter notebooks and may require light editing (e.g. custom paths to data not contained in the repo).

If you would like to details on accessing some external data, please [contact me](#contact). For example, the training dataset for the neural network would be too large to contain in this repository (please see the paper for details on reproducing the training data).

## Contributing

This repo is not intended to be regularly maintained but contributions are welcome. I also encourage you to raise an Issue if you have a question or concern about the work.

## Paper

Lyttle et al. (2021; [preprint](https://arxiv.org/abs/2105.04482))

## Contact

The preferred way to contact me with issues directly related to the code in this repository is by [submitting an issue](https://github.com/alexlyttle/hierarchically-modelling-dwarfs/issues). Otherwise, see the contact information below:

Name: Alex Lyttle

Email: ajl573@student.bham.ac.uk

GitHub: [alexlyttle](https://github.com/alexlyttle)

Twitter: [@_alexlyttle](https://twitter.com/_alexlyttle)

## Acknowledgments

This work is a part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (CartographY; grant agreement ID 804752).
