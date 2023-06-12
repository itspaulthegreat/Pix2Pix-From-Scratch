import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,encode=True,act= "relu",dropout = False):
        super(self,Block).__init__()

        self.Conv =  nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,2,1,padding_mode="reflect",bias=False) if encode else nn.ConvTranspose2d(in_channels,out_channels,4,2,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )

        self.use_dropout = dropout
        self.Dropout  = nn.Dropout(0.5)
    def forward(self,x):
        x = self.Conv(x)
        return self.Dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self,in_channels = 3 ,features = 64):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,features,4,2,1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.down1 = Block(features,features*2,encode=True,act= "leaky",dropout = False)
        self.down2 = Block(features*2,features*4,encode=True,act= "leaky",dropout = False)
        self.down3 = Block(features*4,features*8,encode=True,act= "leaky",dropout = False)
        self.down4 = Block(features*8,features*8,encode=True,act= "leaky",dropout = False)
        self.down5 = Block(features*8,features*8,encode=True,act= "leaky",dropout = False)
        self.down6 = Block(features*8,features*8,encode=True,act= "leaky",dropout = False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1),
            nn.ReLU()
        )

        self.up1 = Block(features*8,features*8,encode=False,act= "relu",dropout = True) # in_channels are multiplied by 2 as it is getting concat with the down data  if u remeber from Unet
        self.up2 = Block(features*8*2,features*8,encode=False,act= "relu",dropout = True)
        self.up3 = Block(features*8*2,features*8,encode=False,act= "relu",dropout = True)
        self.up4 = Block(features*8*2,features*8,encode=False,act= "relu",dropout = True)
        self.up5 = Block(features*8*2,features*8,encode=False,act= "relu",dropout = True)
        self.up6 = Block(features*8*2,features*4,encode=False,act= "relu",dropout = True)
        self.up7 = Block(features*4*2,features*2,encode=False,act= "relu",dropout = True)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2*2,features,4,2,1),
            nn.Tanh()
        )
       
    def forward(self,x):

        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.concat([up1,d7],dim=1))
        up3 = self.up3(torch.concat([up2,d6],dim=1))
        up4 = self.up4(torch.concat([up3,d5],dim=1))
        up5 = self.up5(torch.concat([up4,d4],dim=1))
        up6 = self.up6(torch.concat([up5,d3],dim=1))
        up7 = self.up7(torch.concat([up6,d2],dim=1))
        final = self.final(torch.concat([up7,d1],dim=1))
        return final