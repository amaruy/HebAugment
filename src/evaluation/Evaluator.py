import pandas as pd
from datasets import load_dataset, concatenate_datasets
import sys
sys.path.append('/home/munz/school/research_methods')
import Pipeline as pl

class Evaluator:
    def __init__(self, model_name, entire_train, train, test, batch_size=1000):
        self.model_name = model_name
        self.entire_train = entire_train
        self.train = train
        self.test = test
        self.batch_size = batch_size
        self.batches = []
        self.get_training_batches()

    def get_training_batches(self):
        small_train_texts = set(self.train['text'])

        def filter_fn(example):
            return example['text'] not in small_train_texts
        
        filtered_train = self.entire_train.filter(filter_fn)

        for i in range(0, len(filtered_train), self.batch_size):
            batch = filtered_train.select(range(i, min(i+self.batch_size, len(filtered_train))))
            self.batches.append(batch)

    def concat_batches(self, i):
        concat_train = self.train
        if i > 0:
            for j in range(i):
                concat_train = concatenate_datasets([concat_train, self.batches[j]])
        return concat_train

    def evaluate(self, num_batches):
        pipeline = pl.Pipeline(self.model_name)
        pipeline.seed_everything()
        train = self.concat_batches(num_batches)
        accuracy, _, time_elapsed = pipeline.run(train, self.test)
        return round(accuracy,4), time_elapsed, len(train)

if __name__ == "__main__":
    pl.Pipeline.seed_everything()
    model_name = 'dicta-il/dictabert'
    dataset = load_dataset('hebrew_sentiment')
    train_size = 150
    train = dataset['train'].shuffle(seed=42).select(range(train_size))
    test = dataset['test']

    evaluator = Evaluator(model_name, dataset['train'], train, test)
    
    num_batches = int(sys.argv[1])
    accuracy, time_elapsed, train_size = evaluator.evaluate(num_batches)

    results_path = r'/home/munz/school/research_methods/increasing_train_results.csv'
    results = pd.DataFrame({'num_batches': [num_batches], 'accuracy': [accuracy], 'time_elapsed': [time_elapsed], 'train_size': [train_size]})
    
    try:
        results.to_csv(results_path, mode='a', header=False, index=False)
    except FileNotFoundError:
        results.to_csv(results_path, index=False)

