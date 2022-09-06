import torch
import torchvision
import matplotlib.pyplot as plt

# Load training data from the mnist dataset. 
def load_training():
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output
    return (x_train, y_train)

    
# Load test data from the mnist dataset. 
def load_test():
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output
    return (x_test, y_test)

def plot_images(plt, model):
    fig = plt.figure("W images")
    ax = []
    for i in range(0, 10):
        ax.append(fig.add_subplot(2, 5, i + 1))
        ax[-1].set_title(f"{i}")
        img = model.W[:, i].reshape(28, 28).detach()
        plt.imshow(img)
        plt.imsave(f'W_{i}.png', img)
    plt.show()
   

class SigmoidModel:

    def __init__(self):
        # Model variables
        self.W = torch.zeros(784, 10).clone().detach().requires_grad_(True)
        self.b = torch.zeros(10).clone().detach().requires_grad_(True)

    def f(self, x):
        return torch.softmax(x @ self.W + self.b, dim=1)

    # Use Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.f(x), y)

    # Determines accury of the model
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


def train_model(model, x_train, y_train):

    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.W, model.b], 0.1)
    for _ in range(1000):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    return model


# Setup model
x_train, y_train = load_training()
model = SigmoidModel()

# Train
train_model(model, x_train, y_train)
plot_images(plt, model)

# Test
x_test, y_test = load_test()
print(model.accuracy(x_test, y_test))
