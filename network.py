import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        self.conv256_128 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        )
        self.conv128_64 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.conv64_32 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv32_16 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv16_8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv8_4 = Sequential(
            nn.LeakyReLU(0.2, inplace=True)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv4_2 = Sequential(
            nn.LeakyReLU(0.2, inplace=True)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv2_1 = Sequential(
            nn.ReLU(inplace=True)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )

        self.conv1_2 = Sequential(
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv2_4 = Sequential(
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.5, inplace=True),
        )
        self.conv4_8 = Sequential(
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.5, inplace=True),
        )
        self.conv8_16 = Sequential(
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.5, inplace=True),
        )
        self.conv16_32 = Sequential(
            nn.ConvTranspose2d(512 * 2, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv32_64 = Sequential(
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(256 * 2, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv64_128 = Sequential(
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(128 * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv128_256 = Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh()
        )
    def forward(self, x):
        out256_128 = self.conv256_128(x)
        out128_64 = self.conv128_64(out256_128)
        out64_32 = self.conv64_32(out128_64)
        out32_16 = self.conv32_16(out64_32)
        out16_8 = self.conv16_8(out32_16)
        out8_4 = self.conv8_4(out16_8)
        out4_2 = self.conv4_2(out8_4)
        out2_1 = self.conv2_1(out4_2)
        out1_2 = self.conv1_2(out2_1)
        out2_4 = self.conv2_4(torch.cat((out1_2, out4_2), 0))
        out4_8 = self.conv4_8(torch.cat((out2_4, out8_4), 0))
        out8_16 = self.conv8_16(torch.cat((out4_8, out16_8), 0))
        out16_32 = self.conv16_32(torch.cat((out8_16, out32_16), 0))
        out32_64 = self.conv32_64(torch.cat((out16_32, out64_32), 0))
        out64_128 = self.conv64_128(torch.cat((out32_64, out128_64), 0))
        out128_256 = self.conv128_256(torch.cat((out64_128, out256_128), 0))

        return out128_256

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.conv256_128 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv128_64 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv64_32 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv32_31 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv31_30 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv256_128(x)
        out = self.conv128_64(out)
        out = self.conv64_32(out)
        out = self.conv32_31(out)
        out = self.conv31_30(out)

        return out