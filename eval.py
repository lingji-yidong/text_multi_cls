import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

class BertMultiClassifier:
    def __init__(self, model_path, tokenizer_path, num_labels):
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    def predict(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).tolist()
        return preds, probs.tolist()

    def save_results(self, input_file_path, output_file_path):
        # Load the input file
        df = pd.read_excel(input_file_path)

        # Get the text column
        text_list = df['text'].tolist()

        # Make predictions
        preds, probs = self.predict(text_list)

        # Add the results to the dataframe
        df['pred'] = preds
        df['prob'] = probs

        # Save the results to the output file
        df.to_excel(output_file_path, index=False)
