from operator import index
import torch
import torch.nn as nn
import numpy as np

class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size, emoji_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, 2)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

# Training data

batch_size = 4
chars = [' ', 'o', 'g', 'd', 'h', 'l', 'f', 'r']
emojies = ['🐸', '🐶']
num_chars = len(chars)
num_emojies = len(emojies)

char_encodings = [[1. if i == j else 0. for j in range(num_chars)] for i in range(num_chars)]
emoji_encodings = [[1. if i == j else 0. for j in range(num_emojies)] for i in range(num_emojies)]

def words_to_char_encodings(word):
    return [char_encodings[chars.index(letter)] for letter in word]

def emoji_to_emoji_encodings(emoji):
    return [emoji_encodings[emojies.index(emoji)] for _ in range(batch_size)]

print(words_to_char_encodings(['dog ', 'frog']))
print(emoji_to_emoji_encodings('🐶'))

x_train = torch.tensor(words_to_char_encodings('dog '))
y_train = torch.tensor(emoji_to_emoji_encodings('🐶'))

print(x_train.shape)
print(y_train.shape)

model = LongShortTermMemoryModel(8, 2)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()


y = model.f(torch.tensor([[char_encodings[0]]]))
y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
print(y)
print(y.argmax(1))
