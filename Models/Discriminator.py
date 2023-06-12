import torch
import torch.nn as nn
import torchvision

class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features = [64,128,256,512]):
        super(Discriminator,self).__init__()
        layers = []
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2,features[0],4,2,1,padding_mode="reflect"), # in_channels * 2 becz we have x(input) and y(output) concatenated
            nn.BatchNorm2d(features[0]),
            nn.LeakyReLU(0.2)
        )

        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                self.CnnBlock(in_channels,feature,stride= 1 if feature == features[-1] else 2)
                          )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels,1,4,1,1,padding_mode="reflect")
        )
        self.model = nn.Sequential(*layers)



    def CnnBlock(self,in_channels,out_channels,stride):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,1,padding_mode="reflect",bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


    def forward(self,x,y):
        x = torch.concat([x,y],dim=1)
        x = self.initial(x)
        x = self.model(x)

        return x
    
    
