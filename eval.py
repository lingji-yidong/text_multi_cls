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

        
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define your validation data and dataloader
val_data = # your validation data
val_dataloader = # your validation dataloader

# Evaluate the model on the validation set
model.eval()
val_loss, val_acc, val_f1 = 0, 0, 0
y_true, y_pred = [], []
with torch.no_grad():
    for batch in val_dataloader:
        inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')
        labels = batch['label']
        outputs = model(**inputs, labels=labels)
        loss, logits = outputs[:2]
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_loss += loss.item()
        val_acc += (torch.argmax(logits, dim=1) == labels).sum().item()

# Compute the classification report
class_names = ['class 0', 'class 1']  # replace with your class names
print(classification_report(y_true, y_pred, target_names=class_names))
 
