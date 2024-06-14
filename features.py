import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import os
from datamodule import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            # Forward pass until the third-to-last layer
            outputs = model(images)
            # Extract features from the third-to-last layer
            features_batch = outputs[:, -3].cpu().numpy()
            features.append(features_batch)
            labels.append(targets.cpu().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels


def main():
    path = './dataset/cifar-10-batches-py/'
    X_train, y_train = CIFAR10(path, group='train')
    X_test, y_test = CIFAR10(path, group='test')

    _, c, h, w = X_train.shape
    X_train = X_train.reshape(-1, h, w, c)
    X_test = X_test.reshape(-1, h, w, c)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10Dataset(X_train, y_train, transform=transform)
    test_dataset = CIFAR10Dataset(X_test, y_test, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet18 = models.resnet18(pretrained=True)

    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    resnet18 = resnet18.to(device)

    X_train_features, y_train = extract_features(resnet18, train_loader, device)
    X_test_features, y_test = extract_features(resnet18, test_loader, device)

    print(X_train_features.shape)
    print(X_test_features.shape)
    print(y_train.shape)
    print(y_test.shape)

    with open('train_features.pkl', 'wb') as f:
        pickle.dump((X_train_features, y_train), f)
    with open('test_features.pkl', 'wb') as f:
        pickle.dump((X_test_features, y_test), f)


if __name__ == '__main__':
    main()