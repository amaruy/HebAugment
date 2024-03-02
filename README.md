# **TODO - Deadline 4/3/24!!** 
## Action Items & Responsibilities

Dont forget citations!

### Abstract
- **Yonatan**: Write the abstract section.

### Introduction
- **Yonatan**: Write the introduction section.

### Background and Related Work
- ~**Yonatan**: reWrite the background section.~
- ~**Yonatan**: reWrite the related work section.~


### Methodology
- ~**Amit**: Rewrite the section about generating data with prompt examples and reference GPT.
- **Meiri**: Rewrite the section about translating data and reference models used.

### Evaluation
- **Amit**: Add data analysis for translated data.
- **Meiri**: Add data analysis for translated data.
- **Noam**: Document results with generated data, translated data, and combination of both. Plot results and runtimes.
- **Meiri**: Write the results section and include a table of the different results.


### Conclusions and Future Work
- **Yonatan**: Write the conclusions and future work section.

### Editing
- **Amit**: Ensure the document is 9 pages and check for cohesion.
- **Meiri**: Ensure all references are properly cited.

### Answers to Reviews
- **Yonatan**: Document and prepare responses to review.

### Presentation
- **Baras**: Outline, build, and allocate sections for the presentation.
- **Amit**: Make it pretty.
- **Ido**: ~Create Diagram of proposed system~
- **All**: Practice the presentation.

### Codebase
- **Noam**: Cleanup codebase, write user guide


# HebAugment: Enhancing Hebrew Sentiment Analysis through Data Augmentation

## Overview

HebAugment is dedicated to advancing the field of Natural Language Processing (NLP) in Hebrew, particularly focusing on sentiment analysis. This project addresses the critical challenge of data scarcity in Hebrew by utilizing innovative data augmentation techniques. Through the strategic use of text translation and generation, HebAugment enriches the training dataset, paving the way for more accurate sentiment analysis models.

## Structure

The repository is organized as follows to facilitate easy navigation and comprehension:

- **/data**: Datasets and preprocessing scripts.
  - **test.csv**: test data.
  - **train.csv**: train data.
  - **/translated**: English data and translations.
  - **/generated**: generated data.
- **/models**: Contains both the baseline and augmented model files.
  - **/baseline**: The initial sentiment analysis models.
  - **/improved**: Models trained on the augmented datasets.
- **/notebooks**: Jupyter notebooks for exploratory data analysis, model training, and evaluation.
- **/src**: Core source code.
  - **/augmentation**: Data augmentation scripts.
  - **/classification**: Sentiment classification algorithms.
  - **/evaluation**: Scripts for model performance evaluation.
- **/experiments**: Experimental configurations and results.
- **/scripts**: Utility scripts for automation and data manipulation.

## Links

For further details, ongoing updates, and collaborative contributions, please visit the following resources:

- **[Overleaf Project](https://www.overleaf.com/project/65dca86edeff0e2765e3fce4)**: Comprehensive documentation and research paper drafts.
- **[Google Docs Workspace](https://docs.google.com/document/d/1DCoGTBqNclhy4I-kD9zNqRHxKJWtVFx1mvK6PzVACow/edit)**: Collaborative space for project planning and discussion.

## Abstract

HebAugment introduces a robust method for improving sentiment classification in Hebrew. By effectively addressing the limitations posed by data scarcity, this project leverages cutting-edge NLP techniques to enhance dataset quality and model performance.

## Methodology

**Data Augmentation**: The project employs a two-pronged augmentation approach, integrating translation and generative language models to enrich the Hebrew dataset for sentiment analysis. This strategy not only expands the dataset size but also diversifies the data, contributing to more robust model training.

- **Translation**: Utilizes translation models to convert English datasets to Hebrew, thereby supplementing the original Hebrew data.
- **Generation**: Employs generative models to create new, synthetic data based on the patterns observed in the existing Hebrew dataset.

**Evaluation**: The effectiveness of the augmentation techniques is rigorously assessed through comparative analysis against baseline models. Evaluation metrics focus on accuracy, efficiency, and the overall impact on model performance.

