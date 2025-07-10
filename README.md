# Overview
This repository contains code and data for inferring time-varying selection coefficients in HIV-1 evolution, as well as scripts for generating figures used in the associated preprint.

### Inferring fitness seascapes from evolutionary histories
Yirui Gao<sup>1</sup>, Brian Lee<sup>1</sup> and John P. Barton<sup>2,3,#</sup>

<sup>1</sup> Department of Physics and Astronomy, University of California, Riverside  
<sup>2</sup> Department of Physics and Astronomy, University of Pittsburgh  
<sup>3</sup> Department of Computational and Systems Biology, University of Pittsburgh School of Medicine  
<sup>#</sup> correspondence to [jpbarton@pitt.edu](mailto:jpbarton@pitt.edu)

The preprint is available at [bioRxiv](https://doi.org/10.1101/2025.06.08.658500).

# Contents

- `Simulation_analyze.ipynb`  
  Scripts for generating and analyzing simulated data.

- `HIV_analyze.ipynb`  
  Scripts for processing and analyzing HIV-1 data.

- `figures.ipynb`  
  Scripts for generating figures included in the preprint.


### Demonstration
The typical workflow consists of the following three steps:

1. Processing sequence data.
   For simulations, this means generating evolutionary trajectories.

2. Solving ODEs to infer time-varying selection coefficients.

3. Analyzing results and plotting figures.

As an example, consider the simple simulation described in Figure 2 of the preprint. Input sequences for this example are located in the `data/simulation/simple/sequences` directory. The inferred selection coefficients are saved in the `data/simulation/simple/output` directory. Different output file suffixes indicate different sets of input parameters.


### Software dependencies
Most analyses are performed in Python, relying on the common Python scientific stack ([Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Scipy](https://scipy.org/), [sympy](https://www.sympy.org/en/index.html) and [Matplotlib](https://matplotlib.org/stable/)). For simulation datasets, analyses run in a few seconds on a M chip Macbook. However, for inference tasks for HIV-1 data with long sequences, runtimes can extend to several hours.

Additionally, we also use C++ code from our [previous work](https://github.com/bartonlab/paper-binary-trait-inference) to infer constant selection coefficients. This code requires a C++11 compiler and the [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/). It has been tested with GSL version 2.6. Compilation usually completes within a few seconds on a typical desktop machine.

### HIV-1 data

Here we combine HIV sequence data from [the Los Alamos National Laboratory HIV Sequence Database](https://www.hiv.lanl.gov/content/index) and immunological data to investigate HIV evolution across 13 individuals. This data is contained in the `data/HIV/` directory.

And we use some processed data from our [previous work](https://github.com/bartonlab/paper-binary-trait-inference).


# License
This repository is dual-licensed under:

- [GPL-3.0](LICENSE-GPL) for the source code.
- [CC0 1.0](LICENSE-CC0) for figures, documentation, and our presentation of the data.