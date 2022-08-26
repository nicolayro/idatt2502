import csv
import torch
import matplotlib.pyplot as plt

x = []
y = []
with open("./day_head_circumference.csv", 'r') as file:
    reader = csv.reader(file)
    next(reader) # Skip first line
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

x_train = torch.tensor(x).reshape(-1, 1)
y_train = torch.tensor(y).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * torch.sigmoid((x @ self.W + self.b)) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.000000001)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01
    optimizer.zero_grad()  # Clear gradients for next step


# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.title("Head circumference based on age in days")
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')

# Setup visual model
step = 10
x = torch.linspace(torch.min(x_train), torch.max(x_train), step) # Some length values
y = torch.empty(step)

# Calculate y values based on x
for i, t in enumerate(x):
    print(t)
    y[i] = model.f(torch.tensor([t])).detach()

plt.plot(x, y, label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()