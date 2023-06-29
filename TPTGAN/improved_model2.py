import torch
import torch.nn as nn

from dual_transf import Dual_Transformer


# from conformer import TSCB


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class Net(nn.Module):
    def __init__(self, L=400, width=64):
        super(Net, self).__init__()
        self.L = L
        self.frame_shift = self.L // 2
        # self.N = 256
        # self.B = 256
        # self.H = 512
        # self.P = 3
        # self.device = device
        self.in_channels = 3
        self.out_channels = 1
        # self.kernel_size = (2, 3)
        # self.elu = nn.SELU(inplace=True)
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.width = width

        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width,
                                  kernel_size=(1, 1))  # [b, 64, 257, num_f]
        self.inp_norm = nn.LayerNorm(201)
        self.inp_prelu = nn.PReLU(self.width)

        self.enc_dense1 = DenseBlock(201, 4, self.width)

        self.enc_norm1 = nn.LayerNorm(201)
        self.enc_prelu1 = nn.PReLU(self.width)

        self.dual_transformer = Dual_Transformer(self.width, self.width, num_layers=4)  # [b, 64, f, num_f]

        self.dec_dense1 = DenseBlock(201, 4, self.width)
        # self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(201)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.complexdec = nn.Sequential(
            DenseBlock(201, 4, self.width),
            nn.LayerNorm(201),
            nn.PReLU(self.width),
            nn.Conv2d(in_channels=self.width, out_channels=2, kernel_size=(1, 1)))

        # self.mask_dec_dense1 = DenseBlock(256, 4, self.width)
        # self.mask_dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        # self.mask_dec_norm1 = nn.LayerNorm(512)
        # self.mask_dec_prelu1 = nn.PReLU(self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))
        self.out_norm = nn.LayerNorm(201)
        self.out_prelu = nn.PReLU(1)

    def forward(self, x):
        # x[b, 2, f, num_f]
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)  # mag[b, 1, num, f_size]
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)  # noisy_phase[b, 1, num， f_size]
        x_in = torch.cat([mag, x], dim=1)  # x_in[b, 3, num, f_size]
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x_in)))  # [b, 3, num, f_size]-->[b, 64, num, f_size]

        out = self.enc_dense1(out)  # [b, 64, num, f_size]
        out = self.enc_prelu1(self.enc_norm1(out))  # [b, 64, num, f_size]

        out = self.dual_transformer(out)  # [b, 64, num, f_size]

        out1 = self.dec_dense1(out)  # [b, 64, num, f_size]
        out1 = self.dec_prelu1(self.dec_norm1(out1))  # [b, 64, num, f_size]

        # mask
        # mask = self.mask_dec_dense1(out)
        # mask = self.mask_dec_prelu1(self.mask_dec_norm1(self.mask_dec_conv1(self.pad1(mask))))
        # out = out1 * mask
        out1 = self.out_prelu(self.out_norm(self.out_conv(out1)))  # [b, 1, num, f_size]
        out1 = out1 * mag  # [b, 1, num, f_size]
        out = self.complexdec(out)

        mag_real = out1 * torch.cos(noisy_phase)
        mag_imag = out1 * torch.sin(noisy_phase)
        final_real = mag_real + out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + out[:, 1, :, :].unsqueeze(1)

        # out = torch.squeeze(out, 1)  # [b, f, num_f]
        # # [b, f, num_f]-->[b, sig_len]
        # out = torch.istft(out, n_fft=self.L, hop_length=self.frame_shift, window=torch.hamming_window(self.L).cuda(), normalized=True)

        return final_real, final_imag   # [b, 1, num, f_size]
