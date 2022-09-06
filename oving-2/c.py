import torch
import matplotlib.pyplot as plt

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

class SigmoidModel:

    def __init__(self):
        # Model variables
        self.W1 = torch.tensor([[-0.9, 0.9], [-0.9, 0.8]], requires_grad=True) # requires_grad enables calculation of gradients
        # [[-0.15], [-0.1]] convergers to or
        self.W2 = torch.tensor([[-.85], [-0.9]], requires_grad=True) # requires_grad enables calculation of gradients
        self.b1 = torch.tensor([[-0.9, 0.9]], requires_grad=True)
        self.b2 = torch.tensor([[-0.5]], requires_grad=True)

    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)  # @ corresponds to matrix multiplication

    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)  # @ corresponds to matrix multiplication

    def f(self, x):
        return self.f2(self.f1(x))

    # Use Cross Entropy
    def loss(self, x, y):
        return -torch.mean(torch.multiply(y, torch.log(self.f(x))) + torch.multiply((1 - y), torch.log(1 - self.f(x))))


model = SigmoidModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W1, model.W2, model.b1, model.b2], 0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Visualize result
fig = plt.figure("XOR")
ax = fig.add_subplot(projection="3d")

x = x_train[:, 0].squeeze()
y = x_train[:, 1].squeeze()
z = y_train[:, 0].squeeze()

# Plot datapoints
ax.scatter(x, y, z, label='$(x^{(i)},y^{(i)}),z^{(i)}$')

step = 10

x = torch.linspace(0.0, 1.0, step) 
y = torch.linspace(0.0, 1.0, step)
z = torch.empty((step, step))

# Calculate z values based on x and y
for i, t1 in enumerate(x):
   for j, t2 in enumerate(y):
        tensor = torch.tensor([t1, t2])
        print(model.f(tensor).detach())
        z[i, j] = model.f(tensor).detach()

print(z)
# Plot plane
x_grid, y_grid = torch.meshgrid(x, y)
ax.plot_wireframe(x_grid, y_grid, z, label='$\\hat y = f(x) = xW+b$', color='yellow', alpha=0.75)

plt.show()
