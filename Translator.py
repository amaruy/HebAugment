import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from deep_translator import GoogleTranslator
import concurrent.futures
from tqdm import tqdm


class Translator:
    def __init__(self, path, model_type="transformer"):
        self.path = path
        self.df = pd.read_csv(path)
        self.df = self.df.dropna()[:15000]
        self.model_type = model_type
        # Check if MPS is available and set the device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        if self.model_type == "transformer":
            # For the Transformer model
            # Ensure sentencepiece is installed for tokenizer compatibility
            AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-he")
            self.translator = pipeline("translation_en_to_he", model="Helsinki-NLP/opus-mt-en-he")
        elif self.model_type == "seq2seq":
            model_name = "Helsinki-NLP/opus-mt-en-he"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        elif self.model_type == "attention":
            self.translator = GoogleTranslator(source='en', target='iw')

    def translate(self, text):
        if self.model_type == "transformer":
            # Translate using the Transformer model
            return self.translator(text, max_length=512)[0]['translation_text']
        elif self.model_type == "seq2seq":
            # Translate using the Seq2Seq model
            inputs = self.tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.generate(inputs, max_length=512)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif self.model_type == "attention":
            # Translate using deep-translator
            return self.translator.translate(text)
        else:
            return "Model type not supported"


def contains_english(text):
    # Regular expression to find English words
    return bool(re.search('[a-zA-Z]', text))


def translate_row(row, translator):
    index, data = row
    english_text = data['text']
    translated_text = translator.translate(english_text)
    hebrew_text = translated_text
    sentiment = data['sentiment']
    return english_text, hebrew_text, sentiment


if __name__ == "__main__":
    # # Set options to display long strings
    # pd.set_option('display.max_colwidth', None)
    # models = ["transformer", "seq2seq"]
    # models = ["seq2seq"]
    # for model in models:
    #     translator = Translator('../train.csv', model_type=model)
    #     print(len(translator.df))
    #     # Create lists to store data
    #     english_texts = []
    #     hebrew_texts = []
    #     sentiments = []
    #
    #     for index, row in translator.df.iterrows():
    #         english_texts.append(row['text'])
    #         translated_text = translator.translate(row['text'])
    #         hebrew_texts.append(translated_text)
    #         sentiments.append(row['sentiment'])
    #         if index == 10000:
    #             break
    #
    #     # Count the number of sentences with English words
    #     count = sum(contains_english(sentence) for sentence in hebrew_texts)
    #
    #     print(f"Number of sentences with English words: {count}")
    #
    #     # Create new DataFrame
    #     new_df = pd.DataFrame({
    #         'English Text': english_texts,
    #         'Hebrew Text': hebrew_texts,
    #         'Sentiment': sentiments
    #     })

    pd.set_option('display.max_colwidth', None)

    models = ["transformer", "seq2seq"]
    for model in reversed(models):
        translator = Translator('../train.csv', model_type=model)
        print(f"Model: {model}, Number of rows in DataFrame: {len(translator.df)}")

        # Using ThreadPoolExecutor to parallelize the translation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = ((row, translator) for row in translator.df.iterrows())
            results = list(tqdm(executor.map(lambda args: translate_row(*args), tasks),
                                total=len(translator.df)))
        # results_filtered = [result for result in results if result is not None]
        results_filtered = [result for result in results if result is not None and not contains_english(result[1])]

        # Unpack results
        english_texts, hebrew_texts, sentiments = zip(*results_filtered)

        # Count the number of sentences with English words
        count = sum(contains_english(sentence) for sentence in hebrew_texts)

        print(f"Number of sentences with English words: {count}")

        # Create new DataFrame
        new_df = pd.DataFrame({
            'English Text': english_texts,
            'Hebrew Text': hebrew_texts,
            'Sentiment': sentiments
        })

        new_df.to_csv(
            f'/Users/ormeiri/Desktop/Homework/research methods/translating/ResearchMethodsProject/HebAugment/data/translated/{model}_translated_no_english.csv',
            index=False)

        # # Print each column's values
        # for column in new_df.columns:
        #     print(f"--- {column} ---")
        #     for value in new_df[column]:
        #         print(value)
        #     print("\n")
        # print(new_df)

        # new_df.to_csv(f'{model}_translated.csv', index=False)

        # print(f'English text: {english_text}')
        # print(f'Hebrew text: {hebrow_text}')
    # translator = Translator('train.csv', model_type="transformer")  # Change model_type as needed
    # english_text = translator.df['text'][0]
    # hebrow_text = translator.translate(english_text)
    # print(f'English text: {english_text}')
    # print(f'Hebrew text: {hebrow_text}')
