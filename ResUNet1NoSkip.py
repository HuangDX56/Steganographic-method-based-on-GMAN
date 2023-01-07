import torch
import torch.nn as nn
import numpy as np

# U_net
class sub_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(sub_conv, self).__init__()

        self.in_ch = in_ch
        self.mid_ch = in_ch
        self.out_ch = out_ch

        self.basic = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_ch),
        )        

        self.relu = nn.LeakyReLU(negative_slope=0.2)

        self.conv = nn.Sequential(
            nn.Conv2d(self.mid_ch, self.out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.out_ch),
            nn.LeakyReLU(negative_slope=0.2)
        ) 


    def forward(self, x):
        #residual = x
        out = self.basic(x)
        
        #out += residual
        out = self.relu(out)

        out = self.conv(out)        

        return out



class sub_deconv(nn.Module):
    def __init__(self, in_ch, out_ch, mid=False):
        super(sub_deconv, self).__init__()

        self.in_ch = in_ch
        self.mid_ch = in_ch // 2
        self.out_ch = out_ch
        
        if mid:
            self.mid_ch = in_ch        
        

        self.basic = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, kernel_size=5, stride=1, padding=2, bias=False), 
            nn.BatchNorm2d(self.mid_ch),
            
        )

        self.relu = nn.ReLU()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.mid_ch, self.out_ch, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )
        
        


    def forward(self, res, x):
        #residual = res
        out = self.basic(x)
        
        #out += residual
        out = self.relu(out)

        out = self.deconv(out)

        return out


class UNet(nn.Module):
    def __init__(self, in_ch=1, ngf=16):
        super(UNet, self).__init__()

        self.in_ch = in_ch

        self.layer1 = sub_conv(in_ch, ngf)
        self.layer2 = sub_conv(ngf, ngf*2)
        self.layer3 = sub_conv(ngf*2, ngf*4)
        self.layer4 = sub_conv(ngf*4, ngf*8)
        self.layer5 = sub_conv(ngf*8, ngf*8)
        self.layer6 = sub_conv(ngf*8, ngf*8)
        self.layer7 = sub_conv(ngf*8, ngf*8)
        self.layer8 = sub_conv(ngf*8, ngf*8)

        self.layer9 = sub_deconv(ngf*8, ngf*8, mid=True)
        self.layer10 = sub_deconv(ngf*16, ngf*8)
        self.layer11 = sub_deconv(ngf*16, ngf*8)
        self.layer12 = sub_deconv(ngf*16, ngf*8)
        self.layer13 = sub_deconv(ngf*16, ngf*4)
        self.layer14 = sub_deconv(ngf*8, ngf*2)
        self.layer15 = sub_deconv(ngf*4, ngf)
        self.layer16 = nn.ConvTranspose2d(ngf*2, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, cover):
        self.cover = cover / 255.0
        x1 = self.layer1(self.cover)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)


        x = self.layer9(x8, x8)
        x = self.dropout(x)
        x = self.layer10(x, torch.cat([x, x7], dim=1))
        x = self.dropout(x)
        x = self.layer11(x, torch.cat([x, x6], dim=1))
        x = self.dropout(x)
        x = self.layer12(x, torch.cat([x, x5], dim=1))
        x = self.layer13(x, torch.cat([x, x4], dim=1))
        x = self.layer14(x, torch.cat([x, x3], dim=1))
        x = self.layer15(x, torch.cat([x, x2], dim=1))
        x = self.layer16(torch.cat([x, x1], dim=1))
        
        
        x = torch.sigmoid(x) - 0.5
        p = torch.relu(x)

        return p

