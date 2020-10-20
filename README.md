# kepler-dwarfs

Using machine learning to hierarchically model the first APOKASC catalogue of *Kepler* dwarf and subgiant stars ([Serenelli et al. 2017](https://ui.adsabs.harvard.edu/abs/2017ApJS..233...23S/abstract)). A new method for constraining a helium enrichment law and mixing-length parameter in the solar neighbourhood.

- [kepler-dwarfs](#kepler-dwarfs)
  - [About](#about)
  - [Requirements](#requirements)
  - [Installation and running the code](#installation-and-running-the-code)
  - [Contributing](#contributing)
    - [Submitting a Pull Request](#submitting-a-pull-request)
  - [Paper](#paper)
    - [Contributing to the paper](#contributing-to-the-paper)
  - [Contact](#contact)
  - [Acknowledgments](#acknowledgments)

## About

This repository is the working notebook for my project concerning the hierarchical modelling of low-mass dwarfs in the solar neighbourhood using the machine learning of stellar models. The repository contains Jupyter notebooks and python scripts written for the project.

## Requirements

Requires Python >= 3.5 and a number of packages found in `kepler-dwarfs/requirements.txt`. An unpublished package called `interstellar` is also used for the work in `kepler-dwarfs/training` and `kepler-dwarfs/modelling`. Whilst this package is not publically available, please contact me for details if you wish to run this code. We aim to publish the package when it is tested, and will rerun this work on the published version.

## Installation and running the code

To download the repository, run the following command in a terminal,

```terminal
git clone https://github.com/alexlyttle/kepler-dwarfs.git
```

Much of the work is written in Jupyter notebooks and may require light editing (e.g. custom paths to data not contained in the repo).

If you would like to request access to the external data, please [contact me](#contact).

## Contributing

Contributions to this work should be submitted as a Pull Request or raised as an Issue.

### Submitting a Pull Request

[Instructions on branching and submitting a PR]

## Paper

The paper source code and PDF is found in `kepler-dwarfs/paper`.

### Contributing to the paper

Contributions to the paper should be made on a separate branch (see [Contributing](#contributing)) as to avoid conflicts, and may be reviewed in a Pull Request before merging. To review and comment on existing work, submit a review on the relevant Pull Request.

The LaTeX journal class is [MNRAS v3.0](https://www.ctan.org/tex-archive/macros/latex/contrib/mnras). General instructions to authors may be found [here]( https://academic.oup.com/mnras/pages/General_Instructions).

BibTeX entries may be added to `kepler-dwarfs/paper/references.bib`. Please avoid conflicts and use existing citations if available. The preferred citation key format is `[auth.auth.ea][year]` - the last name of the first two authors (capitalised first letters) and ".ea" if there are more than two, followed by the year of publication.

## Contact

The preferred way to contact me with issues directly related to the code in this repository is by [submitting an issue](https://github.com/alexlyttle/kepler-dwarfs/issues). Otherwise, see the contact information below:

Email: ajl573@student.bham.ac.uk

GitHub: [alexlyttle](https://github.com/alexlyttle)

Twitter: [@_alexlyttle](https://twitter.com/_alexlyttle)

## Acknowledgments

This work is a part of the European Research Council funded project *CartographY*.
