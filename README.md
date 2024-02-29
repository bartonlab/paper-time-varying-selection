
# Overview

This repository contains and data and scripts for reproducing the results accompanying the manuscript  

### Title of paper
Author One<sup>1</sup>, Author Two<sup>2</sup>, and John P. Barton<sup>1,2,#</sup>

<sup>1</sup> Department of Physics and Astronomy, University of California, Riverside  
<sup>2</sup> Department of Computational and Systems Biology, University of Pittsburgh School of Medicine  
<sup>#</sup> correspondence to [jpbarton@pitt.edu](mailto:jpbarton@pitt.edu)  

The preprint is available at __INSERT LINK HERE__.

# Contents

Describe the contents of the repository, and which pieces do what. You can use code text to refer to specific files or directories, like this: `a_file.ipynb`, `a_folder/`. In this template, the `figures.ipynb` contains a template Jupyter notebook for reproducing the figures accompanying the paper. Generally, the generated figures should be placed in the `figures/` directory.

If the analysis uses data that is maintained by a third party or stored separately (e.g., on Zenodo), then it can be linked to here. If the paper develops a method, then a test script should be included that implements the method, allowing users to verify the results before applying the method to their own data.

In general, local data should be organized in the `data/` directory and appropriate sub-folders. This should include raw data that we use (including from simulations) and processed data, which is saved separately.

Significant code should be in the `src/` directory. Code that is just used for making figures or minor data processing could be included here in the top-level directory.

Paper drafts, including cover letters, etc., can be placed in the `drafts/` directory. This directory also contains a `.gitignore` file that will make it such that the contents of this directory are not synced with GitHub.

To sync the above folders on GitHub, placeholder files have been placed in the `data/`, `figures/`, and `src/` directories. These files should be deleted when the template is copied for a project and files appear in each folder.

### Software dependencies

Here's an example statement about the need for external software to execute any part of the code: Parts of the analysis are implemented in C++11 and the [GNU Scientific Library](https://www.gnu.org/software/gsl/).

# License

This repository is dual licensed as [GPL-3.0](LICENSE-GPL) (source code) and [CC0 1.0](LICENSE-CC0) (figures, documentation, and our presentation of the data).
