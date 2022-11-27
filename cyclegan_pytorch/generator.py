import torch
import torch.nn as nn
from residual_block import ResidualBlock


class GeneratorResNet(nn.Module):
	def __init__(self, input_shape, num_residual_block):
		super(GeneratorResNet, self).__init__()

		channels = input_shape[0]

		# Initial Convolution Block
		out_features = 64
		model = [
			nn.ReflectionPad2d(channels),
			nn.Conv2d(channels, out_features, 7),
			nn.InstanceNorm2d(out_features),
			nn.ReLU(inplace=True)
		]
		in_features = out_features

		# Downsampling
		for _ in range(2):
			out_features *= 2
			model += [
				nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
				nn.InstanceNorm2d(out_features),
				nn.ReLU(inplace=True)
			]
			in_features = out_features

		# Residual blocks
		for _ in range(num_residual_block):
			model += [ResidualBlock(out_features)]

		# Upsampling
		for _ in range(2):
			out_features //= 2
			model += [
				nn.Upsample(scale_factor=2),  # --> width*2, heigh*2
				nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
				nn.ReLU(inplace=True)
			]
			in_features = out_features

		# Output Layer
		model += [nn.ReflectionPad2d(channels),
					nn.Conv2d(out_features, channels, 7),
					nn.Tanh()
					]

		# Unpacking
		self.model = nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)
