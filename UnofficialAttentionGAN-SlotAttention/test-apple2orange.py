# For plotting
import numpy as np
import matplotlib.pyplot as plt

# For utilities
import time, sys, os
sys.path.insert(0, '../../')

# For conversion
import cv2
import opencv_transforms.transforms as TF
import dataloader

# For everything
import torch
import torch.nn as nn
import torchvision.utils as vutils

# For our model
import mymodels
import torchvision.models
import itertools

# To ignore warning
import warnings
warnings.simplefilter("ignore", UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device=='cuda':
    print("The gpu to be used : {}".format(torch.cuda.get_device_name(0)))
else:
    print("No gpu detected")

# batch_size. number of cluster
batch_size = 1

# Validation 
print('Loading Validation data...', end=' ')
val_transforms = TF.Compose([
    TF.Resize(128),
    ])

val_imagefolder = dataloader.PairImageFolder('/datasets/CycleGan/apple2orange',
                                              val_transforms,
                                              mode='test')

val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=batch_size, shuffle=True)

print("Done!")
print("Validation data size : {}".format(len(val_imagefolder)))

temp_batch_iter = iter(val_loader)

temp_batch = next(temp_batch_iter)
imgA = temp_batch[0]
imgB = temp_batch[1]

#dataloader.show_example([imgA, imgB], (20,10))


def load(netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D):
    print('Loading...', end=' ')
    checkpoint = torch.load('/kuacc/users/ebostanci18/comp411/AttentionGAN/checkpoint/apple2orange/ckpt.pth')
    netG_A2B.load_state_dict(checkpoint['netG_A2B'], strict=True)
    netG_B2A.load_state_dict(checkpoint['netG_B2A'], strict=True)
    netD_A.load_state_dict(checkpoint['netD_A'], strict=True)
    netD_B.load_state_dict(checkpoint['netD_B'], strict=True)
    optimizer_G.load_state_dict(checkpoint['optimizer_G']),
    optimizer_D.load_state_dict(checkpoint['optimizer_D']),
    print("Done!")
    return checkpoint['epoch']

# A : Edge, B : Color
num_att=5
netG_A2B = mymodels.Generator(input_nc=3, 
                              output_nc=3, 
                              num_att=num_att, 
                              ngf=64, 
                              n_middle=9, 
                              norm='IN', 
                              activation='relu',
                              pretrained=False).to(device)
netG_B2A = mymodels.Generator(input_nc=3, 
                              output_nc=3, 
                              num_att=num_att, 
                              ngf=64, 
                              n_middle=9, 
                              norm='IN', 
                              activation='relu',
                              pretrained=False).to(device)

netD_A = mymodels.Discriminator(input_nc=3, 
                                norm='IN', 
                                activation='lrelu', 
                                pretrained=False).to(device)
netD_B = mymodels.Discriminator(input_nc=3, 
                                norm='IN', 
                                activation='lrelu', 
                                pretrained=False).to(device)

torch.backends.cudnn.benchmark = True


epoch_num = 50
# learning rate
lr = 2e-4
# Loss functions
criterion_L1 = torch.nn.L1Loss() # L1 Loss
# Gamma
gamma = 10
# Lambda
lambda1 = 100
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.999
critic_iter = 1

# Setup Adam optimizers for both G and D
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=lr, betas=(beta1, beta2))

# epoch_num
current_epoch=load(netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D)
print(f"epoch: {current_epoch}")

image_count = 0
def save_example(tensor_list, size, image_iter, image_count):
    n = len(tensor_list)
    plt.figure(figsize=size)
    plt.subplots_adjust(hspace=0, wspace=0)

    for i in range(1, n+1):
        ax1 = plt.subplot(1, n, i)
        result =torch.cat([tensor_list[i-1]],dim=-1)
        plt.imshow(np.transpose(vutils.make_grid(result, nrow=1, padding=5, normalize=True).cpu(),(1,2,0)), aspect='auto')
        plt.axis("off")
    
    plt.savefig(f'/kuacc/users/ebostanci18/comp411/AttentionGAN/results/apple2orange/i{image_iter}-{image_count}.png', bbox_inches='tight')

temp_batch_iter=iter(val_loader)

netG_A2B.eval()
netG_B2A.eval()
temp_batch = next(temp_batch_iter)

for i, data in enumerate(val_loader, 0):
    with torch.no_grad():
    
        imgA = data[0].to(device)
        imgB = data[1].to(device)

        fakeA, _, attB = netG_B2A(imgB)
        fakeB, _, attA = netG_A2B(imgA)

        save_example([imgA, fakeB.detach()], (20, 10), i, image_count)
        image_count += 1
        save_example(attA, (50, 10), i, image_count)
        image_count += 1
        save_example([imgB, fakeA.detach()], (20, 10), i, image_count) 
        image_count += 1
        save_example(attB, (50, 10), i, image_count)
        print(f"{i} done")
        image_count = 0



print('Done')
