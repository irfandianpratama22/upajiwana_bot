import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
import nltk
from nltk_module import bag_of_words, tokenize, stem  # Pastikan Anda menyimpan fungsi ini dalam file nltk_utils.py

# Unduh data NLTK untuk tokenisasi (hanya perlu dijalankan sekali)
nltk.download("punkt")

# Memuat intents dari file JSON
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

all_words = []
tags = []
xy = []

# Loop melalui setiap intent untuk mengekstrak data
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    for pattern in intent['patterns']:
        # Tokenisasi setiap kata dalam kalimat
        words = tokenize(pattern)
        
        # Tambahkan kata ke list all_words
        all_words.extend(words)
        
        # Tambahkan pasangan (pattern, tag) ke dalam xy
        xy.append((words, tag))

# Stemming dan menghapus kata yang tidak diinginkan
ignore_words = ["?", "!", ".", ",", "'"]
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Menghilangkan duplikat dan mengurutkan kata-kata
all_words = sorted(set(all_words))

# Mengurutkan tag secara unik
tags = sorted(set(tags))

print(f'{len(xy)} patterns')
print(f'{len(tags)} tags: {tags}')
print(f'Stemmed words: {len(all_words)}')

# Membuat data pelatihan
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words untuk setiap kalimat pattern
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    # y: Indeks tag dalam daftar tags
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001

input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

print(f'Input size: {input_size}, Output size: {output_size}')

# Dataset untuk pelatihan
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

# Set device (CPU karena tidak menggunakan GPU)
device = torch.device('cpu')

# Model, loss function, dan optimizer
model = NeuralNet(input_size, hidden_size, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Simpan model setelah pelatihan
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')
