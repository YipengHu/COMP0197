# train script
# adapted from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


import os
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()

dataroot = "data"


## Networks ==================================================
nc = 3 # Number of channels in the training images. For color images this is 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    # --- create the two networks
    netG = Generator()
    if use_cuda:
        netG.cuda()
    netG.apply(weights_init)

    netD = Discriminator()
    if use_cuda:
        netD.cuda()
    netD.apply(weights_init)



    ## Dataset =======================================================
    workers = 2
    batch_size = 128
    image_size = 64  #All images will be resized to this

    dataset = torchvision.datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Plot some training images
    real_batch = next(iter(dataloader))
    im = Image.fromarray((np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)).numpy()*255).astype(np.uint8))
    im.save("train_pt_images.jpg")
    print('train_pt_images.jpg saved.')



    ## Losses and optimisers =======================================
    lr = 0.0002  # Learning rate
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

    criterion = nn.BCELoss()
    optimizerD =  torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG =  torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



    ## Training =====================================================
    num_epochs = 5  # Number of training epochs

    manualSeed = 90
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    print("Starting Training Loop...")
    fixed_noise = torch.randn(64, nz, 1, 1)
    if use_cuda:
        fixed_noise = fixed_noise.cuda()
    iters = 0
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            real_images = data[0]
            label = torch.full((real_images.size(0),), 1., dtype=torch.float) #real label
            noise = torch.randn(real_images.size(0), nz, 1, 1)
            if use_cuda:
                real_images, noise, label = real_images.cuda(), noise.cuda(), label.cuda()

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            fake = netG(noise)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label.fill_(0.)) #fake label
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            output = netD(fake).view(-1)
            errG = criterion(output, label.fill_(1.)) #real label
            errG.backward()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[Epoch=%d/%d][%d/%d]\tD-Loss=%.4f\tG-Loss=%.4f\t'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item()))

            # save images
            if (iters%500==0) or ((epoch==num_epochs-1) and (i==len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                im = np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)).numpy()
                Image.fromarray((im*255).astype(np.uint8)).save("gen_pt_images_e%04d_i%06d.jpg" % (epoch,i))

            iters += 1

    print('Training done.')
