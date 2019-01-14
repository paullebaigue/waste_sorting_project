"""https://pytorch.org/docs/stable/nn.html"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def pytorch_tensors():
    x1 = torch.empty(5, 3)
    print("torch.empty \n", x1)
    x2 = torch.rand(5, 3)
    print("torch.rand \n", x2)
    x3 = torch.zeros(5, 3, dtype=torch.long)
    print("torch.zeros \n", x3)
    x4 = torch.tensor([5.5, 3])
    print("torch.tensor([5.5, 3]) \n", x4)
    x5 = x4.new_ones(6, 4, dtype=torch.double)      # new_* methods take in sizes
    print("x4.new_ones(6,4) \n", x5,x4)
    x6 = torch.randn_like(x5, dtype=torch.float)    # override dtype!
    print(x6)
    print("size x6,x5 : \n", x6.size(), x5.size())
    y = torch.rand(6, 4)
    print("x6 + y \n", x6 + y)

# pytorch_tensors()


def pytorch_autograd():
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    y = x + 2
    print(y)

    print("\n y.grad_fn \n", y.grad_fn)
    print("\n y.grad \n", y.grad)

    z = y * y * 3
    out = z.mean()
    print(z, out)

    print(y.requires_grad)
    out.backward()
    print(x.grad, y.grad, z.grad)


def first_network():

    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    # --> Applies a filter (kernel) to the incoming data

    # torch.nn.Linear(in_features=2, out_features=2, bias=True)
    # --> Applies a linear transformation to the incoming data :    y = x . W' + b

    # torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # If the size is a square you can only specify a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)

            # x.view to reshape the tensor
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        # the backward function is automatically defined

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    for k in range(len(params)):
        print(params[k].size())

    print("\n try on a random sample:")
    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    print(output)

    # Every time a variable is back-propogated through, the gradient will be accumulated instead of being replaced
    net.zero_grad()
    output.backward(torch.randn(1, 10), retain_graph=True)

    #computation of the loss
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print("\n MSE = ", loss)

    print("\n backward:")
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    # Backprpoagation of the error
    net.zero_grad()  # zeroes the gradient buffers of all parameters
    print('\n conv1.bias.grad before backward')
    print(net.conv1.bias.grad)
    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # in SGD :     weight = weight - learning_rate * gradient
    ### first way : coding
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    ### second way : package optim
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update


first_network()
