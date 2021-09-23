import torch

print(torch.__version__)

###################################################################################
# 2. Daten einlesen

from datetime import timedelta, date
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

PATH = 'imgs/small'
D = 32


def read_jpg(path):
    '''liest ein JPEG ein und gibt ein DxDx3-Numpy-Array zurück.'''
    img = Image.open(path)
    w, h = img.size
    # schneide etwas Rand ab.
    img = img.crop((5, 24, w - 5, h - 24))
    # skaliere das Bild
    img = img.resize((D, D), Image.ANTIALIAS)
    img = np.asarray(img)
    return img


def read_panitz(directory):
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = date(2010, 10, 30)
    end_date = date(2019, 1, 1)

    imgs = []

    for date_ in daterange(start_date, end_date):
        img_path = '%s/small-b%s.jpg' % (directory, date_.strftime("%Y%m%d"))
        if os.path.exists(img_path):
            img = read_jpg(img_path)
            imgs.append(img)

    return np.array(imgs)


imgs = read_panitz(PATH)

print('Dimension der gelesenen Bilder:', imgs.shape)

# zeigt ein Bild
# plt.imshow(imgs[17])
# plt.show()


###################################################################################
# Hilfsfunktion zum Plotten von Bildern

import math


def plot_reconstructions(imgs, recs, iteration=None):
    print("Plotting...")
    len(imgs)
    # Erstellt ein NxN-Grid zum Plotten der Bilder
    N = int(np.ceil(math.sqrt(2 * len(imgs))))
    f, axarr = plt.subplots(nrows=N, ncols=N, figsize=(18, 18))
    # Fügt die Bilder in den Plot ein
    for i in range(min(len(imgs), 100)):
        axarr[2 * i // N, 2 * i % N].imshow(imgs[i].reshape((D, D, 3)),
                                            interpolation='nearest')
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].imshow(recs[i].reshape((D, D, 3)),
                                                        interpolation='nearest')
    f.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('plots/recs-%.4d.png' % iteration)
    plt.close()

##################################################
# Bilder in ein eindimensionales Array umwandeln

import torchvision

pic_amount = 1854  # Panitz hat 1854 Selfies
t = torch.tensor(imgs)
picSize = D * D * 3
size = len(imgs) * picSize

t = torchvision.transforms.functional.to_tensor(
    imgs.reshape((len(imgs), picSize))
)
t = t[0]

##################################################
# Erstellen der Netzstrucktur und Training

import torch.nn as nn
from random import randint

num_epochs = 10001
learning_rate = 0.001


class PanitzNet(nn.Module):

    def __init__(self):
        super(PanitzNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(picSize, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Linear(100, 50)
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(True),
            nn.Linear(100, 1000),
            nn.ReLU(True),
            nn.Linear(1000, picSize),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def run_training(net, pics):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    pics = pics.cuda()
    for epoch in range(num_epochs):
        # forward
        outputs = net(pics)
        loss = criterion(outputs, pics)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        print("Durchlauf: [", epoch + 1, "von", num_epochs, "] loss:", float(loss))

        if epoch % 1000 == 0:

            recs = outputs.cpu().data
            rand_nums = 10
            x = np.empty((rand_nums, picSize))
            y = np.empty((rand_nums, picSize))

            for i in range(rand_nums):
                rand_num = randint(0, pic_amount - 1)
                x[i] = t[rand_num]
                y[i] = recs[rand_num]

            plot_reconstructions(x, y, epoch)

            torch.save(net.state_dict(), 'pths/panitznet' + str(epoch) + '.pth')


# define the network+data, and train.
net = PanitzNet().cuda()
run_training(net, t)
