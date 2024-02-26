# Evaluator Documentation

This folder contains a Python script for evaluating the performance of machine learning models on NLP tasks with an emphasis on training size variability. The script integrates custom training pipelines and dynamically evaluates models with increasing training set sizes.

## Overview

The `Evaluator` class is designed to systematically assess how varying sizes of training data affect model performance, specifically focusing on sequence classification tasks. It leverages the `Pipeline` class from an external script for model training and evaluation, facilitating a modular and reusable codebase.

### Key Features:

- **Dynamic Training Data Assembly**: Constructs training datasets of increasing size by concatenating predefined batches to the initial training set, allowing for the evaluation of model performance against training size.
- **Batch Management**: Manages training data in batches, enabling efficient data handling and experimentation with different training set sizes without reloading the entire dataset.
- **Performance Evaluation**: Utilizes a customized training pipeline to train models with the assembled training data and evaluates their performance on a test set.

### Dependencies:

- pandas
- datasets (Hugging Face)
- A custom `Pipeline` class for model training and evaluation (expected to be in the system path)

## Usage

To use this script, ensure all dependencies are installed and the `Pipeline` class is accessible in your environment. Customize the `model_name`, `batch_size`, and training/test datasets as needed.

### Example:

```python
model_name = 'dicta-il/dictabert'
dataset = load_dataset('hebrew_sentiment')
train_size = 150
train = dataset['train'].shuffle(seed=42).select(range(train_size))
test = dataset['test']

evaluator = Evaluator(model_name, dataset['train'], train, test)

num_batches = 5  # Specify the number of batches to include in training
accuracy, time_elapsed, train_size = evaluator.evaluate(num_batches)

results_path = '/path/to/save/results/increasing_train_results.csv'
results = pd.DataFrame({
    'num_batches': [num_batches], 
    'accuracy': [accuracy], 
    'time_elapsed': [time_elapsed], 
    'train_size': [train_size]
})

results.to_csv(results_path, mode='a', header=False, index=False)
