import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
import numpy as np
import imageio

torch.manual_seed(1)  # reproducible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
    y = torch.sin(x) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    
    val_x = torch.unsqueeze(torch.linspace(10, 20, 1000), dim=1)
    val_y = torch.sin(val_x)

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y)
    plt.figure(figsize=(10, 4))
    plt.scatter(x.data.numpy(), y.data.numpy(), color="blue")
    plt.title('Regression Analysis')
    plt.xlabel('Independent varible')
    plt.ylabel('Dependent varible')
    plt.savefig('curve_2.png')
    plt.show()

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

        def forward(self, x):
            x = F.relu(self.hidden(x))  # activation function for hidden layer
            x = self.predict(x)  # linear output
            x = torch.sin(x)
            return x

    # another way to define a network
    # net = torch.nn.Sequential(
    #     torch.nn.Linear(1, 200),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(200, 100),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(100, 1),
    # )
    net = Net(1, 200, 1)

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 256
    EPOCH = 10

    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, )
        
    torch_val_dataset = Data.TensorDataset(val_x, val_y)
    val_loader = Data.DataLoader(
        dataset=torch_val_dataset,
        batch_size=1,
        shuffle=False, num_workers=2, )

    my_images = []
    fig, ax = plt.subplots(figsize=(16, 10))
    
    def test(step):
        net.eval()
        tmp = 0
        a = []
        b = []
        c = []
        for step, (batch_x, batch_y) in enumerate(val_loader):
            b_x = Variable(batch_x).to(device)
            b_y = Variable(batch_y).to(device)

            prediction = net(b_x)  # input x and predict based on x
            loss = loss_func(prediction, b_y)
            tmp += loss.cpu().data.numpy().mean()
            a.append(torch.squeeze(b_x.cpu().data).item())
            b.append(torch.squeeze(b_y.cpu().data).item())
            c.append(torch.squeeze(prediction.cpu().data).item())

        plt.cla()
        ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
        ax.set_xlabel('Independent variable', fontsize=24)
        ax.set_ylabel('Dependent variable', fontsize=24)
        ax.set_xlim(9.0, 22.0)
        ax.set_ylim(-1.1, 1.2)
        ax.scatter(a, b, color="blue", alpha=0.2)
        ax.scatter(a, c, color='green', alpha=0.5)
        ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
                fontdict={'size': 24, 'color': 'red'})
        ax.text(8.8, -0.95, 'Loss = %.4f' % loss.cpu().data.numpy(),
                fontdict={'size': 24, 'color': 'red'})

        # Used to return the plot as an image array
        # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        my_images.append(image)

    # start training
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step

            b_x = Variable(batch_x).to(device)
            b_y = Variable(batch_y).to(device)

            prediction = net(b_x)  # input x and predict based on x

            loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if step == 1:
                test(step)
            if step == 899:
                # plot and show learning process
                plt.cla()
                ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
                ax.set_xlabel('Independent variable', fontsize=24)
                ax.set_ylabel('Dependent variable', fontsize=24)
                ax.set_xlim(-11.0, 22.0)
                ax.set_ylim(-1.1, 1.2)
                ax.scatter(b_x.cpu().data.numpy(), b_y.cpu().data.numpy(), color="blue", alpha=0.2)
                ax.scatter(b_x.cpu().data.numpy(), prediction.cpu().data.numpy(), color='green', alpha=0.5)
                ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
                        fontdict={'size': 24, 'color': 'red'})
                ax.text(8.8, -0.95, 'Loss = %.4f' % loss.cpu().data.numpy(),
                        fontdict={'size': 24, 'color': 'red'})

                # Used to return the plot as an image array
                # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
                fig.canvas.draw()  # draw the canvas, cache the renderer
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                my_images.append(image)

    # save images as a gif
    imageio.mimsave('./curve_2_model_3_batch.gif', my_images, fps=12)

    fig, ax = plt.subplots(figsize=(16, 10))
    plt.cla()
    ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-11.0, 13.0)
    ax.set_ylim(-1.1, 1.2)
    ax.scatter(x.data.numpy(), y.data.numpy(), color="blue", alpha=0.2)
    # prediction = net(x)  # input x and predict based on x
    # ax.scatter(x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
    # plt.savefig('curve_2_model_3_batches.png')
    plt.show()