import torch
from torch import nn
from torch.nn.functional import relu, max_pool2d, log_softmax
from torchvision import datasets, transforms
from torch.utils import data 
import argparse
import matplotlib.pyplot as plt

SEED = 0
BATCH_SIZE = 16
EPOCH = 10
TEST_IDX = 42


# fetch dataset
def fetch_MNIST():
    # use pytorch datasets to download MNIST dataset for training and testing
    # create an iterable
    dataset_train = datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())
    loader_train = data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    dataset_test = datasets.MNIST(root="./", train=False, download=True, transform=transforms.ToTensor())
    loader_test = data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    return dataset_train, loader_train, dataset_test, loader_test

# fetch a single item from dataset
def fetch_MNIST_single(dataset, index):
    image, label = dataset[index]
    # transform image shape from (1, 28, 28) to (1, 1, 28, 28) to match input shape of model
    image = torch.unsqueeze(image, dim=0)

    return image, label

# print a single image
def print_MNIST_single(image):
    # transform image shape from (1, 1, 28, 28) to (28, 28) to plot
    image = torch.squeeze(image)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()

# train
def train(model, device, loader_train, optimizer, loss_fn, epoch):
    model.train() # sets the model to training mode to specify behaviour for certain modules which change depending on mode (https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train)
    for batch_idx, (image, label) in enumerate(loader_train):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad() # zero the gradients
        output = model(image) # get model output tensor
        loss = loss_fn(output, label) # calculate error between output and label
        loss.backward() # perform backward pass, calculating gradients for every parameter (with requires_grad=True)
        optimizer.step() # update all parameters with gradients

        # print out loss at every 500 batches
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(loader_train.dataset),
                100. * batch_idx / len(loader_train), loss.item()))

# test
def test(model, device, loader_test, loss_fn):
    model.eval() # same reason as above for train
    test_loss = 0
    correct = 0
    with torch.no_grad(): # disables all gradient calculations changing requires_grad=True to requires_grad=False (https://pytorch.org/docs/stable/generated/torch.no_grad.html)
        for image, label in loader_test:
            image = image.to(device)
            label = label.to(device)
            output = model(image) # get model output tensor
            test_loss += loss_fn(output, label) # accumulate calculated error between output and label
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item() # add to correct if pred is equal to label

    # print loss and number of correct predictions
    test_loss /= len(loader_test.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader_test.dataset),
        100. * correct / len(loader_test.dataset)))
    
    return test_loss

# ConvNet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Conv2d(in_channel, out_channel, kernel_size, stride, padding) (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout(p) (https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Linear(in_feature, out_feature) (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x): # batch size = 32 (defined above)
        x = self.conv1(x) # (32, 1, 28, 28) -> (32, 32, 26, 26)
        x = relu(x)
        x = self.conv2(x) # (32, 32, 26, 26) -> (32, 64, 24, 24)
        x = relu(x)
        x = max_pool2d(x, 2) # (32, 64, 24, 24) -> (32, 64, 12, 12)
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # (32, 64, 12, 12) -> (32, 9216)
        x = self.fc1(x) # (32, 9216) -> (32, 128)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x) # (32, 128) -> (32, 10)
        output = log_softmax(x, dim=1)
        return output

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint", help="checkpoint path") # specify checkpoint to go straight to evaluation
    args = argparser.parse_args()

    # choose device for hardware acceleration
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # random number(s) from seed
    torch.manual_seed(SEED)
    
    _, loader_train, dataset_test, loader_test = fetch_MNIST() # fetch dataset
    model = ConvNet().to(device) # instantiate model on device
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # specify optimizer
    loss_fn = torch.nn.CrossEntropyLoss() # specify loss function

    if args.checkpoint is not None: # check if checkpoint is specified
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint) # load model with checkpoint
        model.eval() # turn on evaluation mode
        image, label = fetch_MNIST_single(dataset_test, TEST_IDX) # fetch single item to evaluate model on
        output = model(image.to(device))
        pred = output.argmax(dim=1).item()
        print(f"predicted: {pred}, label: {label}")
        print_MNIST_single(image)
        return 

    # training and testing for EPOCH epochs
    for epoch in range(1, EPOCH+1):
        train(model=model, device=device, loader_train=loader_train, optimizer=optimizer, loss_fn=loss_fn, epoch=epoch)
        test(model=model, device=device, loader_test=loader_test, loss_fn=loss_fn)
    
    # save model
    print("saving model...")
    torch.save(model.state_dict(), "./mnist.pt")

if __name__ == "__main__":
    main()