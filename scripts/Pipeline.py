import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import os
import gc
import sys

class Pipeline:
    def __init__(self, model_name, epochs=5, batch_size=32, learning_rate=5e-5, max_length=512):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)
        
    @staticmethod
    def seed_everything(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def seconds_to_mm_ss(seconds):
        minutes = '0' + str(int(seconds // 60)) if seconds // 60 < 10 else str(int(seconds // 60))
        seconds = str(round(seconds % 60, 2)) if seconds % 60 >= 10 else '0' + str(round(seconds % 60, 2))
        return minutes + ":" + seconds

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
    
    def get_dataloaders(self, train, test):
        train = train.map(self.tokenize, batched=True)
        test = test.map(self.tokenize, batched=True)
        train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def train_model(self, train_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        progress_bar = tqdm(total=self.epochs * len(train_loader), desc='Training', position=0)
        for epoch in range(self.epochs):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)
                output = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(output.logits, label)
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
        return self.model

    def evaluate_model(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)
                output = self.model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(output.logits, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        
        accuracy = correct / total
        return accuracy

    def run(self, train, test, verbose=True):
        if verbose:
            print('Loading model and tokenizer...')
        start_time = time()

        if verbose:
            print('Getting dataloaders...')
        train_loader, test_loader = self.get_dataloaders(train, test)

        if verbose:
            print('Training model...')
        self.train_model(train_loader)

        if verbose:
            print('Evaluating model...')
        accuracy = self.evaluate_model(test_loader)

        time_elapsed = self.seconds_to_mm_ss(time() - start_time)
        if verbose:
            print(f'Accuracy: {accuracy:.3f}, Time elapsed: {time_elapsed}')
        return accuracy, self.model, time_elapsed
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    Pipeline.seed_everything(42)

    model_name = 'dicta-il/dictabert'

    dataset = load_dataset('hebrew_sentiment')

    train_size = 150
    train = dataset['train'].shuffle(seed=42).select(range(train_size))
    test = dataset['test']

    pipeline = Pipeline(model_name)
    accuracy, model, time_elapsed = pipeline.run(train, test)

    method = 'baseline'
    model_path = '/home/munz/school/research_methods/models/' + method + '.pt'
    pipeline.save_model(model_path)

