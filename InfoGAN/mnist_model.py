import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self):
		super().__init__()

		self.tconv1 = nn.ConvTransposed2d(74, 1024, 1, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(1024)

		self.tconv2 = nn.ConvTransposed2d(1024, 128, 7, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)

		self.tconv3 = nn.ConvTransposed2d(128, 64, 4, 2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(64)

		self.tconv4 = nn.BatchNorm2d(64, 1, 4, 2, 1, padding=1, bias=False)

	def forward(self, x):
		x = torch.relu(self.bn1(self.tconv1(x)))
		x = torch.relu(self.bn2(self.tconv2(x)))
		x = torch.relu(self.bn3(self.tconv3(x)))

		img = torch.sigmoid(self.tconv4(x))

		return img

