![Travis (.org)](https://img.shields.io/travis/HenryDayHall/jetTools)
![Codecov](https://img.shields.io/codecov/c/gh/HenryDayHall/jetTools)
# Tools for jet physics

This repo contains tools for doing jet physics.
Mostly for clustering jets, and a bit for tagging them.

You can find an example of reading `.hepmc` files and `.root` files in `tree_tagger/example.py`.

## Installation

Download and enter the repo:

`git clone https://github.com/HenryDayHall/jetTools.git`

`cd jetTools`

Create a conda environment (optional):

`conda env create -f environment.yml`

`conda activate jet-tools`

Install the package using pip:

`pip install .`

If you'd like to edit the code, you can also do this in development mode:

`pip install -e .`
