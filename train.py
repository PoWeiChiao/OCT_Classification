import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.DenseNet import DenseNet
from model.ResNet import BasicBlock, BottleNeck, ResNet
from utils.dataset import OCTDataset
from utils.logger import Logger

def train(net, device, dataset_train, dataset_val, batch_size=4, epochs=20, lr=0.00001):
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    log_train = Logger('log_train.txt')
    log_val = Logger('log_val.txt')

    valid_loss_min = np.Inf

    for epoch in range(epochs):
        loss_train = 0.0
        loss_val = 0.0
        print('running epoch: {}'.format(epoch))
        net.train()
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            loss_train += loss.item() * image.size(0)

            loss.backward()
            optimizer.step()

        net.eval()
        for image, label in tqdm(val_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            loss_val += loss.item() * image.size(0)

        loss_train = loss_train / len(train_loader.dataset)
        loss_val = loss_val / len(val_loader.dataset)

        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(loss_train, loss_val))
        log_train.write_line(str(epoch) + ',' + str(round(loss_train, 6)))
        log_val.write_line(str(epoch) + ',' + str(round(loss_val, 6)))

        if loss_val <= valid_loss_min:
            torch.save(net.state_dict(), 'model.pth')
            valid_loss_min = loss_val
            print('model saved')

        if epoch >= 10:
            torch.save(net.state_dict(), 'model_' + str(epoch) + '.pth')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ' + str(device))

    net = DenseNet(in_channels=1, num_classes=4)
    # net = ResNet(in_channel=1, n_classes=4, block=BottleNeck, num_block=[3, 4, 6, 3])
    if os.path.isfile('model.pth'):
        net.load_state_dict(torch.load('model.pth', map_location=device))
    net.to(device=device)

    data_dir = 'data'
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset_train = OCTDataset(data_dir, 'train', classes, image_transforms)
    dataset_val = OCTDataset(data_dir, 'val', classes, image_transforms)

    train(net=net, device=device, dataset_train=dataset_train, dataset_val=dataset_val)

if __name__ == '__main__':
    main()