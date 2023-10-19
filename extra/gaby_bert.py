import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertModel, BertTokenizer

class SequencePredictionBERT(torch.nn.Module):
    def __init__(self, model_name, tokenizer):
        super(SequencePredictionBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 1)  # Output 1 value
        self.tokenizer = tokenizer
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        predicted_value = self.linear(sequence_output).squeeze(-1)
        return predicted_value

class CustomDictionaryDataset(Dataset):
    def __init__(self, data_dict, tokenizer):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        target_value = self.keys[index]
        input_values_str = self.data_dict[target_value].split()
        input_values = [int(value) for value in input_values_str]
        target = int(target_value)
        
        input_text = " ".join(input_values_str)  # Join input values as text
        input_encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        return input_encoding, target

# Load data from CSV and preprocess
data = pd.read_csv("data.csv")

# Extract the 'y' column and convert it to a list
y_column = data['y'].tolist()
c_y = [item.strip('[]') for item in y_column]
# Preprocess the X column values

X_column = data['X'].tolist()
X_list = [item.strip("[]") for item in X_column]
X_list = [item.split() for item in X_list]
X_final = []
for row in X_list:
    X_caca = " ".join(row)
    X_final.append(X_caca)
dic = {key.lstrip(): value.lstrip() for key, value in zip(c_y, X_final)}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
custom_dataset = CustomDictionaryDataset(dic, tokenizer)

batch_size = 64
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

model = SequencePredictionBERT("bert-base-uncased", tokenizer)
criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.eval()
# Input sequence
input_sequence = [6, 6, 6, 5, 5, 4, 4, 2, 1, 1, 1, 1, 1, 6, 12]

# Convert the input sequence to text and encode it
input_text = " ".join(str(value) for value in input_sequence)
input_encoding = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    padding='max_length',
    max_length=512,
    return_tensors='pt'
)

# Make predictions
with torch.no_grad():
    input_ids = input_encoding['input_ids']
    attention_mask = input_encoding['attention_mask']
    predicted_sequence = model(input_ids, attention_mask)
    predicted_class = predicted_sequence.argmax().item()

print("Predicted Next Sequence:",  predicted_class)
