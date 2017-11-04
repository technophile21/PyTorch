
# coding: utf-8

# In[1]:


from __future__ import print_function


# In[2]:


import torch as tc
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


class EncoderStack(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1):
        super(EncoderStack, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding)
        self.maxPool = nn.MaxPool2d(2, 2)

    def forward(self, inp):
        print("Encoder: {}".format(inp.size()))
        x = self.conv1(inp)
        print("Encoder conv1: {}".format(x.size()))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        conv_saved = x
        x = self.maxPool(x)
        print("Encoder maxpool: {}".format(x.size()))
        return x, conv_saved


# In[4]:


class DecoderStack(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, upsample_size=None, padding=0, stride=1):
        super(DecoderStack, self).__init__()
        
        self.upsample = nn.Upsample(upsample_size, scale_factor=2, mode='bilinear')
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding)


    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        print("Upsampled: {}".format(upsampled.size()))
        print("Bypass: {}".format(bypass.size()))
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
        
        print("Bypassed: {}".format(bypass.size()))
        
        return tc.cat((upsampled, bypass), 1)

    def forward(self, inp, bypass):
        print("Dec input: {}".format(inp.size()))
        #Upsample doesn't seem necessary as we have stride = 2 for up conv 
        '''
        x = self.upsample(inp)
        print("Dec upsample: {}".format(x.size()))
        '''
        x = self.deconv(inp)
        print("Dec deconv: {}".format(x.size()))
        x = self._crop_concat(x, bypass)
        print("Dec concat: {}".format(x.size()))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x



class UNet(nn.Module):
    def __init__(self, in_shape, num_classes):
        super(UNet, self).__init__()

        channels, width, height = in_shape

        self.num_classes = num_classes
        
        self.enc1 = EncoderStack(3, 64, 3)
        self.enc2 = EncoderStack(64, 128, 3)
        self.enc3 = EncoderStack(128, 256, 3)
        self.enc4 = EncoderStack(256, 512, 3)
        #possible dropout layer here
        
        self.droplayer = nn.Dropout()
        
        self.center = nn.Sequential(
            #Maybe we need just two conv here??
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.dec1 = DecoderStack(1024, 512, 3)
        self.dec2 = DecoderStack(512, 256, 3)
        self.dec3 = DecoderStack(256, 128, 3)
        self.dec4 = DecoderStack(128, 64, 3)

        self.conv = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, inp):
        x, enc_saved1 = self.enc1(inp)
        print("Enc1: {}".format(x.size()))
        x, enc_saved2 = self.enc2(x)
        print("Enc2: {}".format(x.size()))
        x, enc_saved3 = self.enc3(x)
        x, enc_saved4 = self.enc4(x)
        print("Enc4: {}".format(x.size()))
        x = self.droplayer(x)
        x = self.center(x)
        print("Center: {}".format(x.size()))
        x = self.dec1(x, enc_saved4)
        print("Dec1: {}".format(x.size()))
        x = self.dec2(x, enc_saved3)
        x = self.dec3(x, enc_saved2)
        x = self.dec4(x, enc_saved1)
        print("Dec4: {}".format(x.size()))
        x = self.conv(x)
        print("Final: {}".format(x.size()))
        return x





