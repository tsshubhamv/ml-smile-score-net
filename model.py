import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

print('torch version: ', torch.__version__)
print('torchvision version: ', torchvision.__version__)


def assertReqLibraries():
    """To check whether using valid version"""
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)


def createDataGenerators(config):
    """To create data generators for training and validation"""
    input_path = config.DATASET_PATH
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize
        ]),
        'validation':
        transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(), normalize]),
    }

    image_datasets = {
        'train':
        datasets.ImageFolder(input_path + 'train', data_transforms['train']),
        'validation':
        datasets.ImageFolder(input_path + 'validation',
                             data_transforms['validation'])
    }

    dataloaders = {
        'train':
        DataLoader(image_datasets['train'],
                   batch_size=32,
                   shuffle=True,
                   num_workers=0),
        'validation':
        DataLoader(image_datasets['validation'],
                   batch_size=32,
                   shuffle=False,
                   num_workers=0)
    }
    dataset_size = {
        'train': image_datasets['train'],
        'validation': image_datasets['validation']
    }
    return dataloaders, dataset_size


def createModel(config, mode='ne'):
    """It creates resnet50 model"""
    model = models.resnet50(pretrained=True).to(config.device)
    fc_layer = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True),
                             nn.Linear(128, 2), nn.Softmax()).to(config.device)
    if mode == 'load':
        model.fc = fc_layer
        artifacts_path = config.ARTIFACTS_PATH
        model.load_state_dict(
            torch.load(os.path.join(artifacts_path,
                                    f"{config.MODEL_NAME}.pth")))

        return model, None, None
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = fc_layer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters())

        return model, criterion, optimizer


def train(model,
          loss_fn,
          optimizer,
          dataloaders,
          dataset_size,
          device,
          num_epochs=100):
    """Train the model"""
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            print('{} loss: {:.4f}, acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    return model


def saveModel(model, config):
    """Saves the model at given path"""
    artifacts_path = config.ARTIFACTS_PATH
    torch.save(model.state_dict(),
               os.path.join(artifacts_path, f"{config.MODEL_NAME}.pth"))


def loadModel(config):
    """Loads saved model and restores weigths"""
    model = createModel(config=config, mode='load')
    return model


def main(config):
    """main function to train or inference"""
    mode = config.MODE
    assertReqLibraries()
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    if mode == 'train':
        dataloaders, dataset_size = createDataGenerators(config=config)
        model, loss_fn, optimizer = createModel(config=config, mode='new')
        trained_model = train(model,
                              loss_fn,
                              optimizer,
                              dataloaders,
                              dataset_size,
                              device=config.device,
                              num_epochs=100)
        saveModel(trained_model, config=config)
    else:
        model = createModel(config=config, mode='load')
