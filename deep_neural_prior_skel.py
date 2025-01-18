import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image 
import torchvision.transforms as transforms 
from matplotlib import pyplot as plt


class MyUNet(nn.Module):
    def __init__(self):
        super(MyUNet, self).__init__()
        #encoder
        self.down1=nn.Sequential(nn.Conv2d(16, 32, 5, 2, 2, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.down2=nn.Sequential(nn.Conv2d(32, 64, 5, 2, 2, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.down3=nn.Sequential(nn.Conv2d(64, 128, 5, 2, 2, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.down4=nn.Sequential(nn.Conv2d(128, 256, 5, 2, 2, bias=False), nn.LeakyReLU(0.2, inplace=True))

        self.skip128= nn.Sequential(nn.Conv2d(128, 4, 5, 1, 2, bias=False),nn.LeakyReLU(0.2, inplace=True))
        self.skip64= nn.Sequential(nn.Conv2d(64, 4, 5, 1, 2, bias=False),nn.LeakyReLU(0.2, inplace=True))

        self.up4=nn.Sequential(nn.ConvTranspose2d(256, 124, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.up3=nn.Sequential(nn.ConvTranspose2d(128, 60, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.up2=nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.up1=nn.Sequential(nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False))
        
    def forward(self,x):
        x1=self.down1(x)
        x2=self.down2(x1)
        x2p=self.skip64(x2)
        x3=self.down3(x2)
        x3p=self.skip128(x3)
        x4=self.down4(x3)
        x5=self.up4(x4)
        x5p=torch.concat([x5,x3p],dim=1)
        x6=self.up3(x5p)
        x6p=torch.concat([x6,x2p],dim=1)
        x7=self.up2(x6p)
        x8=self.up1(x7)
        x9=torch.sigmoid(x8)
        return x9
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

myunet = MyUNet()
myunet.to(device)


input_img = Image.open('testcrop.jpg')
transform = transforms.Compose([transforms.PILToTensor()])
target = transform(input_img)/255
target = target.unsqueeze(0).float().to(device)

h = target.size()[2]
w = target.size()[3]
z = torch.rand(1,16, h, w).to(device)


#Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(myunet.parameters(), lr=0.001) 
losslog = []

#for inpainting



#for denoising


output = myunet(z)
#test
for i in range(0,2000):
    optimizer.zero_grad() 
    loss=criterion(output, target)
    loss.backward()
    losslog.append(loss.item())
    optimizer.step()
    if i%10 == 0:
        print(f"epoch {i}, Loss : ", loss)

#save the model
torch.save(myunet, "weights.pth")
    
plt.imsave('rebuild.jpg', output[0].cpu().detach().permute(1,2,0).numpy())

# display the loss
#plt.figure(figsize=(6,4))
#plt.yscale('log')
#plt.plot(losslog, label = 'loss ({:.4f})'.format(losslog[-1]))
#plt.xlabel("Epochs")
#plt.legend()
#plt.show()
#plt.close()


# #version qui marche
# import numpy as np
# import torch
# from torch import nn
# from torch import optim
# import torch.nn.functional as F
# from PIL import Image 
# import torchvision.transforms as transforms 
# from matplotlib import pyplot as plt
# import random

# class MyUNet(nn.Module):
#     def __init__(self):
#         super(MyUNet, self).__init__()
#         # down layers
#         self.conv_down_1 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
#         self.conv_down_2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
#         self.conv_down_3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
#         self.conv_down_4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
#         # skip layers
#         self.conv_skip_1 = nn.Conv2d(128, 4, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv_skip_2 = nn.Conv2d(64, 4, kernel_size=5, stride=1, padding=2, bias=False)
#         # up layers
#         self.conv_up_4 = nn.ConvTranspose2d(256, 124, kernel_size= 4, stride=2, padding=1)
#         self.conv_up_3 = nn.ConvTranspose2d(128, 60, kernel_size= 4, stride=2, padding=1)
#         self.conv_up_2 = nn.ConvTranspose2d(64, 32, kernel_size= 4, stride=2, padding=1)
#         self.conv_up_1 = nn.ConvTranspose2d(32, 3, kernel_size= 4, stride=2, padding=1)
    
#         self.model = nn.ModuleList([
#             self.conv_down_1,
#             self.conv_down_2,
#             self.conv_down_3,
#             self.conv_down_4,
#             self.conv_skip_1,
#             self.conv_skip_2,
#             self.conv_up_4,
#             self.conv_up_3,
#             self.conv_up_2,
#             self.conv_up_1
#         ])

#     def forward(self,x):
#         x = nn.functional.leaky_relu(self.conv_down_1(x))
#         conv2_activated = nn.functional.leaky_relu(self.conv_down_2(x))
#         conv3_activated = nn.functional.leaky_relu(self.conv_down_3(conv2_activated))
#         x = nn.functional.leaky_relu(self.conv_down_4(conv3_activated))
#         x = torch.concat([nn.functional.leaky_relu(self.conv_up_4(x)), nn.functional.leaky_relu(self.conv_skip_1(conv3_activated))], dim=1)
#         x = torch.concat([nn.functional.leaky_relu(self.conv_up_3(x)), nn.functional.leaky_relu(self.conv_skip_2(conv2_activated))], dim=1)
#         x = nn.functional.leaky_relu(self.conv_up_2(x))
#         x = nn.functional.leaky_relu(self.conv_up_1(x))
#         x = torch.sigmoid(x)
#         return x
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# myunet = MyUNet()
# myunet.to(device)

# input_img = Image.open('testcrop.jpg')
# transform = transforms.Compose([transforms.PILToTensor()])
# target = transform(input_img)/255
# target = target.unsqueeze(0).float().to(device)

# h = target.size()[2]
# w = target.size()[3]
# z = torch.rand(1,16, h, w).to(device)

# #Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(myunet.model.parameters(), lr=0.001)
# losslog = []

# loadWeightsFromFile = False
# noisyImage = True

# if loadWeightsFromFile:
#     print("Loading weights from weights.pth")
#     torch.load(myunet, "weights.pth")

# if noisyImage:
#     _, height, width = target.shape
#     num_noisy_pixels = 5000
#     for _ in range(num_noisy_pixels):
#         # Sélectionner des coordonnées aléatoires pour un pixel
#         x = random.randint(0, height - 1)
#         y = random.randint(0, width - 1)    
#         # Remplacer les valeurs RGB du pixel par [0, 0, 0] (noir)
#         target[0, :, x, y] = torch.tensor([0.0, 0.0, 0.0])

# image = transforms.ToPILImage()(target.squeeze(0))  # Enlever la dimension batch
# image.show()

# #test

# for i in range(0,2000):
#     optimizer.zero_grad()
#     output = myunet(z)
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()
#     if i%10 == 0:
#         print(f"epoch {i}, Loss : ", loss)

# output = myunet(z)

# print("saving model in weights.pth")
# torch.save(myunet, "weights.pth")

# plt.imsave('final.jpg', output[0].cpu().detach().permute(1,2,0).numpy())

# # display the loss
# plt.figure(figsize=(6,4))
# plt.yscale('log')
# plt.plot(losslog, label = 'loss ({:.4f})'.format(losslog[-1]))
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()
# plt.close()