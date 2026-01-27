import torch
import torch.nn as nn

class FF(nn.Module):
	def __init__(self, input_width, output_width, bias=True):
		super().__init__()
		self.linear_in = nn.Linear(input_width, output_width, bias=bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
	
		output = self.linear_in(x)  # (batch_size, parallel_size * n_nodes)  # (batch_size, output_width)
		return output