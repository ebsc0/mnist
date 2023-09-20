import torch
from torch import nn
from torch.nn.functional import relu, max_pool2d, log_softmax
from torchvision import datasets, transforms
from torch.utils import data 
import argparse

SEED = 1
BATCH_SIZE = 32
EPOCH = 3

torch.manual_seed(SEED)

# fetch dataset
def fetch_MNIST():
    dataset_train = datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())
    loader_train = data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    dataset_test = datasets.MNIST(root="./", train=False, download=True, transform=transforms.ToTensor())
    loader_test = data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    return loader_train, loader_test

# train
def train(model, loader_train, optimizer, loss_fn, epoch):
    model.train()
    for batch_idx, (image, label) in enumerate(loader_train):
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(loader_train.dataset),
                100. * batch_idx / len(loader_train), loss.item()))

# eval
def test(model, loader_test, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in loader_test:
            output = model(image)
            test_loss += loss_fn(output, label)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(loader_test.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader_test.dataset),
        100. * correct / len(loader_test.dataset)))
    
    return test_loss

# ConvNet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = log_softmax(x, dim=1)
        return output

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint", help="checkpoint path")
    args = argparser.parse_args()

    loader_train, loader_test = fetch_MNIST()
    model = ConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.checkpoint is not None:
        model.eval()
        checkpoint = torch.load(args.checkpoint)
        model.state_dict(checkpoint)

        for image, label in loader_test:
            output = model(image)
            print(f"pred: {output}, label: {label}")
            break
        return

    for epoch in range(1, EPOCH+1):
        train(model=model, loader_train=loader_train, optimizer=optimizer, loss_fn=loss_fn, epoch=epoch)
        test(model=model, loader_test=loader_test, loss_fn=loss_fn)
    
    print("saving model...")
    torch.save(model.state_dict(), "./mnist.pt")

if __name__ == "__main__":
    main()