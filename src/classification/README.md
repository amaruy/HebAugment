# Machine Learning Pipeline Documentation

This folder contains a comprehensive Python script designed for training, evaluating, and saving machine learning models, specifically focused on sequence classification tasks using transformers. The script is built to be adaptable for various NLP tasks and datasets.

## Overview

The core of this script is the `Pipeline` class, which encapsulates the entire process of model training and evaluation. It is designed to work with text data, leveraging the Hugging Face `transformers` and `datasets` libraries for ease of use and flexibility.

### Key Features:

- **Model Training and Evaluation**: The pipeline allows for the training of sequence classification models using predefined datasets. It evaluates the model's performance in terms of accuracy on a specified test set.
- **Tokenizer and Model Loading**: Utilizes the `AutoTokenizer` and `AutoModelForSequenceClassification` for dynamic loading of tokenizers and models based on the user's input.
- **Customizable Training Parameters**: Users can specify model name, number of training epochs, batch size, learning rate, and the maximum token length for the input sequences.
- **Device Management**: Automatically detects and utilizes GPU resources if available, falling back to CPU otherwise.
- **Utility Functions**: Includes functions for seeding (for reproducibility), time measurement, tokenization, dataloader preparation, model saving, and more.
- **Model Saving**: Offers functionality to save the trained model for later use or inference.

### Dependencies:

- pandas
- numpy
- datasets (Hugging Face)
- transformers (Hugging Face)
- torch
- tqdm

## Usage

To use this script, ensure all dependencies are installed. You can then customize the `model_name`, `epochs`, `batch_size`, `learning_rate`, and `max_length` parameters as needed for your specific task.

### Example:

```python
model_name = 'dicta-il/dictabert'
dataset = load_dataset('hebrew_sentiment')

train_size = 150
train = dataset['train'].shuffle(seed=42).select(range(train_size))
test = dataset['test']

pipeline = Pipeline(model_name)
accuracy, model, time_elapsed = pipeline.run(train, test)

method = 'baseline'
model_path = '/path/to/save/model/' + method + '.pt'
pipeline.save_model(model_path)
