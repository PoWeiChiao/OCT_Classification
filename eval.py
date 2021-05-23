import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.DenseNet import DenseNet
from model.ResNet import BasicBlock, BottleNeck, ResNet
from utils.dataset import OCTDataset
from utils.logger import Logger

def predict(net, device, dataset_test):
    test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

    results = np.zeros((4, 4))

    net.eval()
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(device=device, dtype=torch.float32)
            pred = net(image)
            pred = np.array(pred.data.cpu()[0])
            results[label[0]][np.argmax(pred)] += 1
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    net = ResNet(in_channel=1, n_classes=4, block=BottleNeck, num_block=[3, 4, 6, 3])
    net.to(device=device)
    net.load_state_dict(torch.load('model.pth', map_location=device))

    data_dir = 'data'
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset_train = OCTDataset(data_dir, 'test', classes, image_transforms)

    results = predict(net, device, dataset_train)
    print(results)

    output = Logger('test_results')
    output.write_line('CNV,' + str(results[0][0]) + ',' + str(results[0][1]) + ',' + str(results[0][2]) + ',' + str(results[0][3]))
    output.write_line('DME,' + str(results[1][0]) + ',' + str(results[1][1]) + ',' + str(results[1][2]) + ',' + str(results[1][3]))
    output.write_line('DRUSEN,' + str(results[2][0]) + ',' + str(results[2][1]) + ',' + str(results[2][2]) + ',' + str(results[2][3]))
    output.write_line('NORMAL,' + str(results[3][0]) + ',' + str(results[3][1]) + ',' + str(results[3][2]) + ',' + str(results[3][3]))

if __name__ == '__main__':
    main()