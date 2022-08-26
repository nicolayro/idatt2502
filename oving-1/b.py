import csv
import torch
import matplotlib.pyplot as plt

# Read data from input
x = []
y = []
with open("./day_length_weight.csv", 'r') as file:
    reader = csv.reader(file)
    next(reader) # Skip first line
    for row in reader:
        y.append(float(row[0]))
        x.append([float(row[1]), float(row[2])])

# Observed/training input and output
x_train = torch.tensor(x).reshape(-1, 2)
y_train = torch.tensor(y).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True) # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001, 0.9)
for epoch in range(100000):
    model.loss(x_train, y_train).backward() # Compute loss gradients
    optimizer.step() # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure("Age of babies based on length and weight")
ax = fig.add_subplot(projection="3d")

length = x_train[:, 0].squeeze()
width = x_train[:, 1].squeeze()
days = y_train[:, 0].squeeze()

# Plot datapoints
ax.scatter(length, width, days, label='$(x^{(i)},y^{(i)}),z^{(i)}$')

step = 30

x = torch.linspace(torch.min(length), torch.max(length), step) # Some length values
y = torch.linspace(torch.min(width), torch.max(width), step) # Some weight values
z = torch.empty((step, step), dtype=torch.long) # Initialize z values

# Calculate z values based on x and y
for i, t1 in enumerate(x):
    for j, t2 in enumerate(y):
        tensor = torch.tensor([t1, t2])
        z[i, j] = model.f(tensor).detach()

# Plot plane
x_grid, y_grid = torch.meshgrid(x, y)
ax.plot_surface(x_grid, y_grid, z, label='$\\hat y = f(x) = xW+b$', color='yellow', alpha=0.75)

plt.show()
