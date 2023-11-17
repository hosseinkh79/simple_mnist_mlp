import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


ROOT_DATA_PATH = 'E:\\Programming\\Per\\Python\\Uni_Projects\\Neural_Networks\\mnist_project\\data'

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray to a torch.FloatTensor
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training set
batch_size = 64

train_dataset = datasets.MNIST(root=ROOT_DATA_PATH, train=True, transform=transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Similarly, you can download and load the test set
test_dataset = datasets.MNIST(root=ROOT_DATA_PATH, train=False, transform=transform, download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)