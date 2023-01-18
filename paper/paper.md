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

Interpreting the results of machine learning algorithms has become an increasingly important topic in the field, as concerns about the lack of interpretability of these models have grown [@Burkart2021]. While many libraries and frameworks provide tools for making predictions and evaluating model performance, it can be challenging to understand how the predictions are being made and the factors that are driving them. This lack of interpretability can make it difficult to trust the results of these models, particularly when they are being used to make important decisions.

Natural sciences, in particular, have seen an increase in interest in the use of machine learning due to its capability of analyzing large amounts of data in a fast and economic way, its generalized application, and easy access through several libraries and products. At the same time, explainable artificial intelligence (XAI) has gained demand in science since results may lack scientific value if no explanation is possible, which can affect scientific consistency going forward and, as mentioned, diminish the significance of the results for each knowledge domain [@Roscher2020].

One simple approach to improve the interpretability of machine learning algorithms is through the use of sensitivity analysis, which involves systematically varying the input features and measuring the resulting change in the prediction. By comparing the predictions obtained with the original values of the features to those obtained with the modified values, it is possible to understand how the prediction changes as each feature is varied. An example of the latter would be RELIEF [@Kira1992], a feature selection method that detects statistical significant features according the change in the target. 

Even though there are products and libraries available for explaining different types of algorithms, such as SHAP [@Lundberg2017], LIME [@Ribeiro2016], or GradCAM [@Selvaraju2017], they can be difficult to implement, interpret, or understand for technical scientists with little or no data science background. As such, a tool with an easy and understandable approach that can help the different stakeholders better understand the algorithm can considerably increase the transparency, comprehensibility, and scientific impact of machine learning results in natural sciences and other applications [@Bhatt2020, @Belle2021].


# Overview

The pudu library is a Python tool that aims to address the challenge of interpretability in machine learning by providing a deterministic and agnostic post-hoc approach by conducting sensitivity analysis on classification and regression tasks. This toolbox uses the same basic principles as RELIEF, and gives ample liberty to the user on its parameters and use cases. pudu quantifies the relevance of features according to changes in the target instance by changing the input in a user-defined way. To perform this, pudu assumes that the algorithm in question has a classification probability function or a prediction function, for classification and regression problems, respectively. With the latter, the features space is changed in a deterministic and sequential way and quantifies the relevance of the features according to the change in the target. As mentioned, both classification and regression problems can be analyzed, and 2-d (vectors) and 3-d (non-RGB images) are accepted as features. pudu develops visual explanations in an easy way by showing the feature relevance on each decision, being also useful to perform local explanations by selecting specific areas of interest.

To illustrate this functionality, examples using scikit-learn [@Pedregosa2011], keras [@chollet2018keras], and localreg are included in the documentation, along with the use of LIME and GradCAM to show how pudu can complement the explainability of AI decisions in different types of problems.

pudu is built in Python 3 [@VanRossumGuidoDrake2009], and also uses third-party packages including numpy [@Harris2020] and matplotlib [@Hunter2007]. pudu comes with complete documentation, including quick start, examples, and contribution guidelines. Source code and documentation can be downloaded from https://github.com/pudu-py/pudu.


# Features

A brief list of features includes:
- A one-fits-all class for every accepted problem and algorithm.
- Importance: estimates the absolute or relative importance oif the features.
- Speed: calculates how fast a prediction changes according to the changes in the features.
- Synergy: tests teh synergy between features and the change in classification probability.
- Easy plotting of the results from the above.


# Acknowledgements

This work has received funding from the European Union's Horizon 2020 Research and Innovation Programme under grant agreement no. 952982 (Custom-Art project) and Fast Track to Innovation Programme under grant agreement no. 870004 (Solar-Win project). Authors from IREC belong to the SEMS (Solar Energy Materials and Systems) Consolidated Research Group of the “Generalitat de Catalunya” (ref. 2017 SGR 862) and are grateful to European Regional Development Funds (ERDF, FEDER Programa Competitivitat de Catalunya 2007–2013). MG acknowledges the financial support from Spanish Ministry of Science, Innovation and Universities within the Juan de la Cierva fellowship (IJC2018-038199-I).


# Authors contribution with [CRediT](/guides/content/editing-an-existing-page)

- Enric Grau-Luque: Conceptualization, Data curation, Software, Writing – original draft
- Ignacio Becerril-Romero: Investigation, Methodology, Writing – review & edition
- Alejandro Perez-Rodriguez: Funding acquisition, Project administration, Resources, Supervision
- Maxim Guc: Formal analysis, Validation, Methodology, Writing – review & edition
- Victor Izquierdo-Roca: Funding acquisition, Project administration, Supervision


# References
