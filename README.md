<img src="https://raw.githubusercontent.com/pudu-py/pudu/main/docs/_static/pudu-baner.png" width="70%">

[![image](https://img.shields.io/pypi/v/pudu.svg)](https://pypi.python.org/pypi/pudu)
[![image](https://img.shields.io/conda/vn/conda-forge/pudu.svg)](https://anaconda.org/conda-forge/pudu)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CodeQL](https://github.com/pudu-py/pudu/actions/workflows/codeql.yml/badge.svg)](https://github.com/pudu-py/pudu/actions/workflows/codeql.yml)
[![image](https://github.com/pudu-py/pudu/workflows/docs/badge.svg)](https://pudu-py.github.io/pudu)
[![codecov](https://codecov.io/gh/pudu-py/pudu/branch/main/graph/badge.svg?token=DC0QIwuYel)](https://codecov.io/gh/pudu-py/pudu)
[![Downloads](https://static.pepy.tech/personalized-badge/pudu?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=pypi%20downloads)](https://pepy.tech/project/pudu)
[![image](https://img.shields.io/conda/dn/conda-forge/pudu?color=blue&label=conda%20downloads)](https://anaconda.org/conda-forge/pudu)
[![image](https://img.shields.io/badge/stackoverflow-Ask%20a%20question-brown?logo=stackoverflow&logoWidth=18&logoColor=white)](https://stackoverflow.com/questions/tagged/pudu)

**A Python library for explainability of machine learinng algorithms in an agnostic, deterministic, and simple way.**

* GitHub repo: https://github.com/pudu-py/pudu
* Documentation: https://pudu-py.github.io/pudu
* PyPI: https://pypi.python.org/pypi/pudu
* Conda-forge: https://anaconda.org/conda-forge/pudu
* Free software: MIT license

# Introduction

**pudu** is a Python package that helps intepret and explore the results of machinme learning algorythms. It does this by quantifying the change
in probability according to the change in the features. This library works with any case that has a probability function, which is normally available in ``scikit-learn`` and ``keras`` methods, and works for both 1-d (vectrors) and 2-d problems (images). In order to see the exact procedure and format needed, please refer to the examples in the ``docs``.

# Features

The following is a list of the main procedures that **pudu** package enables.

- Importance: estimates the absolute or relative importance oif the features.
- Speed: calculates how fast a preditci0on changes according to the changes in the features.
- Synergy: tests teh synergy between features and the change in classification probability.
- Easy plotting of the results from the above.

# Quickstart

1. Install this library using ``pip``:

        pip install pudu

2. Install this library using ``conda-forge``:

        conda install -c conda-forge pudu

3. Test it by running one of the examples in the ``docs``.

4. If you find this library useful, please consider a reference or citation as:

        ...

5. Stay up-to-date by updating the library using:

       conda update pudu
       pip install --update pudu

6. If you encounter problems when updating, try uninstalling and then re-installing::

        pip uninstall pudu
        conda remove pudu


# Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the
[giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
