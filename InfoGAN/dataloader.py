import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

# Directory containing the data.
root = 'data/'

def get_data(dataset, batch_size):

    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root, train='train', 
                                download=True, transform=transform)

    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = dsets.SVHN(root, split='train', 
                                download=True, transform=transform)

    elif dataset = 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root, train='train', 
                                download=True, transform=transform)

    elif dataset = 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    return dataloader