import torch.nn as nn
import math
import torch


def conv3x3(in_planes, out_planes, stride = 1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class down(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size = 3, stirde = 2, mode = 'conv'):
		super(down, self).__init__()
		if mode == 'conv':
			self.down = nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, 
						stride = 2, padding = 1, bias = False)
		elif mode == 'pool':
			self.down = nn.MaxPool2d(2)

	def forward(self, x):
		x = self.down(x)

		return x


class up(nn.Module):
	def __init__(self, in_planes, out_planes, bilinear = True):
		super(up, self).__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor = 2)
		else:
			self.up = nn.ConvTranspose2d(in_planes, out_planes, 2, stride = 2)

	def forward(self, x):
		x = self.up(x)

		return x


class WaveBlock(nn.Module):
	expansion = 1
	def __init__(self, in_ch, out_ch, stride = 1):
		super(WaveBlock, self).__init__()

		self.down = down(in_ch, out_ch, mode = 'conv')
		self.up = up(out_ch, out_ch)
		self.conv = conv3x3((in_ch + out_ch), out_ch, stride = stride)
		self.bn = nn.BatchNorm2d(out_ch)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		x1 = x

		x = self.down(x)
		x = self.up(x)

		x = torch.cat([x1, x], dim = 1)
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)

		return x


class WaveNet(nn.Module):
	def __init__(self, in_ch, block, layers, num_classes = 2):
		super(WaveNet, self).__init__()

		self.in_planes = 64
		self.conv1 = nn.Conv2d(in_ch, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace = True)
		self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)

		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

		self.avgpool = nn.AvgPool2d(2, stride = 1, padding = 0)
		self.fc = nn.Linear(4608, num_classes)


	def _make_layer(self, block, planes, blocks, stride = 1):
		layers = []
		layers.append(block(self.in_planes, planes, stride))
		self.in_planes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_planes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def wavenet18(in_ch, **kwargs):
	model = WaveNet(in_ch, WaveBlock, [2, 2, 2, 2], **kwargs)
	return model

def wavenet34(in_ch, **kwargs):
	model = WaveNet(in_ch, WaveBlock, [3, 4, 6, 3], **kwargs)
	return model























