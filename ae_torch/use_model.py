# coding: utf-8

from model import Pre_dataset
from model import ConvAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load("test.pt"))
    model.eval()

    batch_size = 1
    test_x = Pre_dataset("dataset_test", transform=transforms.ToTensor())

    testx_loader = DataLoader(test_x, batch_size=batch_size, shuffle=False)

    print(len(test_x))
    criterion = nn.BCELoss()

    for idx, (x, y) in enumerate(testx_loader):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        # print(outputs[0])
        img = np.transpose(outputs[0].cpu().detach().numpy(), (1, 2, 0))
        print(criterion(outputs, y))
        plt.imshow(img)
        plt.show()
        # save_image(outputs[0], f"{idx}.png")


