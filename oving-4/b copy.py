from operator import index
import torch
import torch.nn as nn
import numpy as np

class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size, emoji_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128, batch_first=True)  # 128 is the state size
        self.dense = nn.Linear(128, emoji_size)  # 128 is the state size

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        out = out[:, -1, :]
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

# Training data

batch_size = 4
chars = [' ', 'o', 'g', 'd', 'h', 'l', 'f', 'r']
emojies = {
    "frog": '🐸', 
    "dog ": '🐶',
}
num_chars = len(chars)
num_emojies = len(emojies)
index_to_emojies = ['🐸', '🐶']

char_encodings = [[1. if i == j else 0. for j in range(num_chars)] for i in range(num_chars)]
emoji_encodings = [[1. if i == j else 0. for j in range(num_emojies)] for i in range(num_emojies)]

def word_to_char_encodings(word):
    return [char_encodings[chars.index(letter)] for letter in word]

def emoji_to_emoji_encodings(emoji):
    return [emoji_encodings[list(emojies.values()).index(emoji)] for _ in range(batch_size)]


model = LongShortTermMemoryModel(8, 2)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for e in emojies.items():
        model.reset(4)
        x_train = torch.tensor(word_to_char_encodings(e[0]))
        y_train = torch.tensor(emoji_to_emoji_encodings(e[1]))
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

model.reset(1)
y = model.f(torch.tensor(word_to_char_encodings('dog ')))
print(y)
print("'dog ': ", index_to_emojies[y.argmax(1)])
model.reset()
y = model.f(torch.tensor(word_to_char_encodings('frog')))
print("'frog' : ", index_to_emojies[y.argmax(1)])
model.reset()
y = model.f(torch.tensor(word_to_char_encodings('frg ')))
print("'frg' : ", index_to_emojies[y.argmax(1)])

model.reset()
y = model.f(torch.tensor(word_to_char_encodings('drog')))
print("'drog' : ", index_to_emojies[y.argmax(1)])
