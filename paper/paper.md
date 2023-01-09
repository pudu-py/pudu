---
title: "pudu: A generalized and agnostic Python library for explainability of Machine Learning classification and regression problems."
tags:
    - Python
    - Machine Learning
    - Explainability and intepretability
    - Combinatorial analysis
    - Classification and regression
authors:
    - name: Enric Grau-Luque
      orcid: 0000-0002-8357-5824
      affiliation: "1"
    - name: Ignacio Becerril-Romero
      orcid: 0000-0002-7087-6097
      affiliation: "1"
    - name: Alejandro Perez-Rodriguez
      orcid: 0000-0002-3634-1355
      affiliation: "1, 2"
    - name: Maxim Guc
      orcid: 0000-0002-2072-9566
      affiliation: "1"
    - name: Victor Izquierdo-Roca
      orcid: 0000-0002-5502-3133
      affiliation: "1"
affiliations:
    - name: Catalonia Institute for Energy Research (IREC), Jardins de les Dones de Negre 1, 08930 Sant Adrià de Besòs, Spain
      index: 1
    - name: Departament d'Enginyeria Electrònica i Biomèdica, IN2UB, Universitat de Barcelona, C/ Martí i Franqués 1, 08028 Barcelona, Spain
      index: 2
date: 09 January 2023
bibliography: paper.bib
---

# Statement of need

Interpreting the results of machine learning algorithms has become an increasingly important topic in the field, as concerns about the lack of interpretability of these models have grown. While many libraries and frameworks provide tools for making predictions and evaluating model performance, it can be challenging to understand how the predictions are being made and the factors that are driving them. This lack of interpretability can make it difficult to trust the results of these models, particularly when they are being used to make important decisions.

One approach to improving the interpretability of machine learning algorithms is through the use of sensitivity analysis, which involves systematically varying the input features and measuring the resulting change in the prediction. By comparing the predictions obtained with the original values of the features to those obtained with the modified values, it is possible to understand how the prediction changes as each feature is varied.


# Overview
The **`pudu`** library is a Python tool that aims to address the challenge of interpretability in machine learning by providing a deterministic and non-random approach for conducting sensitivity analysis on classification and regression tasks. In this paper, we present three examples to demonstrate the utility of **`pudu`** for interpreting the results of machine learning algorithms.

**`pudu`** is built in Python 3 [@VanRossumGuidoDrake2009], and also uses third-party packages including `numpy` [@Harris2020], `pandas` [@Reback2021], and `matpotlib` [@Hunter2007]. **`pudu`** comes with complete documentation, including quick start, examples, and contribution guidelines. Source code and documentation can  be
downloaded from https://github.com/pudu-py/pudu.


# Features

A brief list of features includes:

- Importance: estimates the absolute or relative importance oif the features.
- Speed: calculates how fast a preditci0on changes according to the changes in the features.
- Synergy: tests teh synergy between features and the change in classification probability.
- Easy plotting of the results from the above.

# Acknowledgements

This work has received funding from the European Union's Horizon 2020 Research and Innovation Programme under grant agreement no. 952982 (Custom-Art project) and Fast Track to Innovation Programme under grant agreement no. 870004 (Solar-Win project). Authors from IREC belong to the SEMS (Solar Energy Materials and Systems) Consolidated Research Group of the “Generalitat de Catalunya” (ref. 2017 SGR 862) and are grateful to European Regional Development Funds (ERDF, FEDER Programa Competitivitat de Catalunya 2007–2013). MG acknowledges the financial support from Spanish Ministry of Science, Innovation and Universities within the Juan de la Cierva fellowship (IJC2018-038199-I).

# References
