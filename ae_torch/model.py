# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import os
from PIL import Image
from torchvision.utils import save_image
from torchsummary import summary


class Pre_dataset(Dataset):
    # im_name_list, resize_dim,
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.im_list = os.listdir(self.root_dir)
        self.inputdir = os.listdir(f"{self.root_dir}/input")
        self.outputdir = os.listdir(f"{self.root_dir}/output")
        # self.resize_dim = resize_dim
        self.transform = transform

    def __len__(self):
        return len(self.inputdir)

    def __getitem__(self, idx):
        # im = Image.open(os.path.join(self.root_dir, self.im_list[idx]))
        input = Image.open(os.path.join(f"{self.root_dir}/input", self.inputdir[idx]))
        output = Image.open(os.path.join(f"{self.root_dir}/output", self.outputdir[idx]))

        input = input.resize((256, 256))
        input = np.array(input)

        output = output.resize((256, 256))
        output = np.array(output)

        if self.transform:
            input = self.transform(input)
            output = self.transform(output)

        return input, output


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model
    model = ConvAutoencoder()
    model.to(device)
    summary(model, [(3, 256, 256)])

    # Loss function
    criterion = nn.BCELoss()

    def my_loss(output, target):
        loss = torch.mean((output - target)**2)
        return loss

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.cuda()
    # summary(model, [(3, 512, 512)])
    batch_size = 5

    train_x = Pre_dataset("dataset", transform=transforms.ToTensor())

    test_x = Pre_dataset("dataset_test", transform=transforms.ToTensor())

    trainx_loader = DataLoader(train_x, batch_size=batch_size, shuffle=True)

    testx_loader = DataLoader(test_x, batch_size=batch_size, shuffle=False)

    n_epochs = 50
    # Training code
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        model.train()
        train_loss = 0.0

        # Training
        for idx, (x, y) in enumerate(trainx_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if idx % 10 == 0:
                print(f"epochs: {epoch}, train_step: {idx / len(trainx_loader)}, train_loss: {train_loss / len(trainx_loader)}")

        train_loss = train_loss / len(trainx_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # torch.save(model.state_dict(), "test1.pt")

