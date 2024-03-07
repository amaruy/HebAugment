We hope to build a API around this project in the future, currently, it is publicly available for research purposes.

# HebAugment: Enhancing Hebrew Sentiment Analysis through Data Augmentation

## Overview

HebAugment is dedicated to advancing the field of Natural Language Processing (NLP) in Hebrew, particularly focusing on sentiment analysis. This project addresses the critical challenge of data scarcity in Hebrew by utilizing innovative data augmentation techniques. Through the strategic use of text translation and generation, HebAugment enriches the training dataset, paving the way for more accurate sentiment analysis models.

## Paper
In the future, we aim to publish our paper [Beyond the Data Scarcity Barrier: Unlocking Sentiment Analysis in Hebrew with Data Augmentation](HebAugment.pdf).

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

## Future Work
- Integrate a in-house LLM with a better hebrew tokenizer than OPEN-AI chatGPT 4
- test on a wide varation of datasets
- modul code

