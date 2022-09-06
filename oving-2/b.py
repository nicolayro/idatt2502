import torch
import matplotlib.pyplot as plt

# Logical nand
x_train = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        ])
y_train = torch.tensor([
    [1.0],
    [1.0],
    [1.0],
    [0.0]
    ])

class SigmoidModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True) # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return -torch.mean(torch.multiply(y, torch.log(self.f(x))) + torch.multiply((1 - y), torch.log(1 - self.f(x))))


model = SigmoidModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(1000):
    if epoch % 100 == 0:
        print("[W]: ", model.W)
        print("[b]: ", model.b)
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure("NAND")
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
        z[i, j] = model.f(tensor).detach()

print(z)
# Plot plane
x_grid, y_grid = torch.meshgrid(x, y)
ax.plot_wireframe(x_grid, y_grid, z, label='$\\hat y = f(x) = xW+b$', color='yellow', alpha=0.75)

plt.show()
