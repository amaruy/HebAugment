# Translation Pipeline Documentation

This folder contains a Python script named `Translator`, designed to translate text from English to Hebrew using different machine learning models. It demonstrates the flexibility of using pre-trained models from the Hugging Face `transformers` library and custom translation logic.

## Overview

The `Translator` class encapsulates the functionality needed to load a dataset from a CSV file, preprocess it by dropping NA values, and perform translations using selected models. It supports multiple translation models, including transformer-based and sequence-to-sequence (Seq2Seq) models.

### Key Features:

- **Flexible Model Selection**: Users can specify the type of model to use for translation (`transformer`, `seq2seq`, or a custom `attention` mechanism model).
- **Automatic Device Detection**: The script automatically detects if Apple's Metal Performance Shaders (MPS) are available for PyTorch, utilizing GPU acceleration when possible.
- **Preprocessing**: Input data is preprocessed to remove NA values, ensuring clean text for translation.
- **Batch Translation**: Although not explicitly batched in the code, the translation process can easily be adapted to handle large datasets efficiently.

### Dependencies:

- pandas
- transformers
- torch
- re (for regular expressions)

## Usage

Before running the script, ensure all dependencies are installed. Adjust the path to your CSV file containing the text to be translated. You can select the model type by changing the `model_type` parameter when initializing the `Translator` object.

### Example:

```python
translator = Translator('path/to/your/data.csv', model_type='seq2seq')
translated_text = translator.translate("Your English text here")
print(translated_text)
