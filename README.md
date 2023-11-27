<div align="center">
<pre>
  __  __ _   _ ___ ____ _____ 
 |  \/  | \ | |_ _/ ___|_   _|
 | |\/| |  \| || |\___ \ | |  
 | |  | | |\  || | ___) || |  
 |_|  |_|_| \_|___|____/ |_|  
</pre>
</div>
Convolutional Neural Network (CNN) working with the MNIST dataset.

# MNIST
Modified from https://github.com/pytorch/examples/blob/main/mnist/main.py

Modified National Institute of Standards and Technology (MNIST) is a dataset of handwritten digits.

The dataset is seperated into 60,000 (training set) and 10,000 (testing set) images.

Each image is a 1x28x28 array representing a black and white (single channel) 28x28 image. 

## Usage
For training and testing
```console
python mnist.py
```
For loading checkpoint
```console
python mnist.py --checkpoint ./mnist.pt
```

## Model
### 2 Convolutional Layers (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
Conv2d is the dot product or the element-wise multiplication and accumulation of the image and kernel.

![2d convolution with padding 1](https://imgs.search.brave.com/07SFpapGm91oSHpayXXmHnUxztz2HEmlHNpybFC9iWw/rs:fit:860:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy8x/LzE5LzJEX0NvbnZv/bHV0aW9uX0FuaW1h/dGlvbi5naWY.gif)

### 2 Dropout Layers (https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
>During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

>This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/pdf/1207.0580.pdf)

### 2 Linear Layers (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

Transforms tensor into linear tensor.

### Max Pooling (https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html#torch.nn.functional.max_pool2d)

Reduces the resolution of the image by selecting the maximum value within a kernel sliding over the input image.

![Pooling techniques](https://imgs.search.brave.com/2fIUR6kxT2j4BmMCbOo0GqvbEKmhre4D5n22xSqsvPc/rs:fit:860:0:0/g:ce/aHR0cHM6Ly93d3cu/Ym91dmV0Lm5vL2Jv/dXZldC1kZWxlci91/bmRlcnN0YW5kaW5n/LWNvbnZvbHV0aW9u/YWwtbmV1cmFsLW5l/dHdvcmtzLXBhcnQt/MS9fL2F0dGFjaG1l/bnQvaW5saW5lL2U2/MGU1NmE2LThiY2Qt/NGI2MS04ODBkLTdj/NjIxZTJjYjFkNTo2/NTk1YTY4NDcxZWQz/NzYyMTczNDEzMGNh/MmNiNzk5N2ExNTAy/YTJiL1Bvb2xpbmcu/Z2lm.gif)

### ReLU

Rectifier Linear Unit is an [activation function](https://ai.stackexchange.com/questions/5493/what-is-the-purpose-of-an-activation-function-in-neural-networks).

![ReLU](https://imgs.search.brave.com/RI-Ve8jQM5ickJzkfCyuN0y3Vk9mP7NxAb01zB4BAqY/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9weXRv/cmNoLm9yZy9kb2Nz/L3N0YWJsZS9faW1h/Z2VzL1JlTFUucG5n)

### Forward Pass
```python
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
```