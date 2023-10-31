---
title: "pudu: A Python library for agnostic feature selection and explainability of Machine Learning spectroscopic problems."
tags:
    - Python
    - Spectroscopy
    - Machine Learning
    - Explainability and intepretability
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
date: 30 June 2023
bibliography: paper.bib
---


# Statement of need

Spectroscopic techniques (e.g. Raman, photoluminescence, reflectance, transmittance, X-ray fluorescence) are an important and widely used resource in different fields of science, such as photovoltaics [@Fonoll-Rubio2022] [@Grau-Luque2021], cancer [@Bellisola2012], superconductors [@Fischer2007], polymers [@Easton2020], corrosion [@Haruna2023], forensics [@Bhatt2023], and environmental sciences [@Estefany2023], to name just a few. This is due to the versatile, non-destructive and fast acquisition nature that provides a wide range of material properties, such as composition, morphology, molecular structure, optical and electronic properties. As such, machine learning (ML) has been used to analyze spectral data for several years, elucidating their vast complexity, and uncovering further potential on the information contained within them [@Goodacre2003] [@Luo2022]. Unfortunately, most of these ML analyses lack further interpretation of the derived results due to the complex nature of such algorithms. In this regard, interpreting the results of ML algorithms has become an increasingly important topic, as concerns about the lack of interpretability of these models have grown [@Burkart2021]. In natural sciences (like materials, physical, chemistry, etc.), as ML becomes more common, this concern has gained especial interest, since results obtained from ML analyses may lack scientific value if they cannot be properly interpreted, which can affect scientific consistency and strongly diminish the significance and confidence in the results, particularly when tackling scientific problems [@Roscher2020].

Even though there are methods and libraries available for explaining different types of algorithms such as SHAP [@Lundberg2017], LIME [@Ribeiro2016], or GradCAM [@Selvaraju2017], they can be difficult to interpret or understand even for data scientists, leading to problems such as miss-interpretation, miss-use and over-trust [@Kaur2020]. Adding this to other human-related issues [@Krishna12022], researchers with expertise in natural sciences with little or no data science background may face further issues when using such methodologies [@Zhong2022]. Furthermore, these types of libraries normally aim for problems composed of image, text, or tabular data, which cannot be associated in a straightforward way with spectroscopic data. On the other hand, time series (TS) data shares similarities with spectroscopy, and while still having specific needs and differences, TS dedicated tools can be a better approach. Unfortunately, despite the existence of several libraries that aim to explain models for TS with the potential to be applied to spectroscopic data, they are mostly designed for a specialized audience, and many are model-specific [@Rojat2021]. Furthermore, spectral data typically manifests as an array of peaks that are typically overlapped and can be distinguished by their shape, intensity, and position. Minor shifts in these patterns can indicate significant alterations in the fundamental properties of the subject material. Conversely, pronounced variations in the other case might only indicate negligible differences. Therefore, comprehending such alterations and their implications is paramount. This is still true with ML spectroscopic analysis where the spectral variations are still of primary concern. In this context, a tool with an easy and understandable approach that offers spectroscopy-aimed functionalities that allow to aim for specific patterns, areas, and variations, and that is beginner and non-specialist friendly is of high interest. This can help the different stakeholders to better understand the ML models that they employ and considerably increase the transparency, comprehensibility, and scientific impact of ML results [@Bhatt2020] [@Belle2021].


# Overview

**pudu** is a Python library that helps to make sense of ML models for spectroscopic data by quantifying changes in spectral features and explaining their effect to the target instances. In other words, it perturbates the features in a predictable and deliberate way and evaluates the features based on how the final prediction changes. For this, four main methods are included and defined. **Importance** quantifies the relevance of the features according to the changes in the prediction. Thus, this is measured in probability or target value difference for classification or regression problems, respectively. **Speed** quantifies how fast a prediction changes according to perturbations in the features. For this, the Importance is calculated at different perturbation levels, and a line is fitted to the obtained values and the slope, or the rate of change of Importance, is extracted as the Speed. **Synergy** indicates how features complement each other in terms of prediction change after perturbations. Finally, **Re-activations** account for the number of unit activations in a Convolutional Neural Network (CNN) that after perturbation, the value goes above the original activation criteria. The latter is only applicable for CNNs, but the rest can be applied to any other ML problem, including CNNs. To read in more detail how these techniques work, please refer to the [definitions](https://pudu-py.github.io/pudu/definitions.html) in the documentation.

pudu is versatile as it can analyze classification and regression algorithms for both 1- and 2-dimensional problems, offering plenty of flexibility with parameters, , and the ability to provide localized explanations by selecting specific areas of interest. To illustrate this, \autoref{fig:figure1} shows two analysis instances using the same `importance` method but with different parameters. Additionally, its other functionalities are shown in examples using scikit-learn [@Pedregosa2011], keras [@chollet2018keras], and localreg [@Marholm2022] are found in the documentation, along with XAI methods including LIME and GradCAM.

**pudu** is built in Python 3 [@VanRossum2009] and uses third-party packages including numpy [@Harris2020], matplotlib [@Caswell2021], and keras. It is available in both PyPI and conda, and comes with complete documentation, including quick start, examples, and contribution guidelines. Source code and documentation are available in https://github.com/pudu-py/pudu.

![Two ways of using the same method *importance* by A) using a sequential change pattern over all the spectral features and B) selecting peaks of interest. In A), the impact of the peak in the range of 1200-1400 opaques the impact of the rest. In contrast, in B) only the first four main peaks are selected to be analyzed and better visualize their impact in the prediction.\label{fig:figure1}](figure1.png)


# Acknowledgements

Co-funded by the European Union (GA No. 101058459 Platform-ZERO). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Health and Digital Executive Agency (HADEA). Neither the European Union nor the granting authority can be held responsible for them. This project has received funding from the European Union's Horizon 2020 research and innovation programme under Marie Skłodowska-Curie GA No. 801342 (Tecniospring INDUSTRY) and the Government of Catalonia's Agency for Business Competitiveness (ACCIÓ). Authors from IREC belong to the MNT-Solar Consolidated Research Group of the “Generalitat de Catalunya” (ref. 2021 SGR 01286) and are grateful to European Regional Development Funds (ERDF, FEDER Programa Competitivitat de Catalunya 2007–2013).

# Authors contribution with [CRediT](https://credit.niso.org/)

- Enric Grau-Luque: Conceptualization, Data curation, Software, Writing – original draft
- Ignacio Becerril-Romero: Investigation, Methodology, Writing – review & edition
- Alejandro Perez-Rodriguez: Funding acquisition, Project administration, Resources, Supervision
- Maxim Guc: Formal analysis, Validation, Methodology, Writing – review & edition
- Victor Izquierdo-Roca: Funding acquisition, Project administration, Supervision

# References
