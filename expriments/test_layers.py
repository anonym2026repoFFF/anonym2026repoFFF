import unittest

import torch

from .new_fff import FFF
from .old_fff import oldFFF
from .moe import MoELayer


class TestFFF(unittest.TestCase):

    def setUp(self):
        self.B = 3
        self.T = 5
        self.d_hidden = 256
        self.d_intermediate = 256

    def test_new_fff(self):
        print('')

        depth = 5
        n_nodes = 2 ** (depth + 1) - 1
        parallel_size = self.d_intermediate // n_nodes
        print(f'depth = {depth}, n_nodes = {n_nodes}, parallel_size = {parallel_size}')

        fff_new = FFF(
            input_width=self.d_hidden,
            output_width=self.d_hidden,
            depth=depth,
            parallel_size=parallel_size,
            activation=torch.nn.GELU,
        )

        total_params = sum([p.numel() for p in fff_new.parameters()])
        print(f'new FFF total_params = {total_params}')

        n_active_nodes = (fff_new.depth + 1) * fff_new.parallel_size
        n_active_node_params = n_active_nodes * fff_new.input_width
        n_output_proj_params = n_active_nodes * fff_new.output_width
        n_active_params = n_active_node_params + n_output_proj_params

        print(f'FFF n_active_params = {n_active_params} ({n_active_params / total_params:.2f})')

        fff_new = torch.compile(fff_new)
        x_in = torch.randn(self.B, self.T, self.d_hidden)
        print(fff_new(x_in).shape)

    def test_old_fff(self):
        print('')

        depth = 3
        leaf_width = self.d_hidden // (2 ** depth)

        old_fff = oldFFF(
            input_width=self.d_hidden,
            output_width=self.d_hidden,
            depth=depth,
            leaf_width=leaf_width,
        )
        old_fff = torch.compile(old_fff)
        x_in = torch.randn(self.B, self.T, self.d_hidden)
        print(old_fff(x_in).shape)

    def test_moe(self):
        print('')

        n_experts = 7
        moe_k = 2
        expert_hidden_dim = self.d_intermediate // n_experts

        moe = MoELayer(
            input_dim=self.d_hidden,
            output_dim=self.d_hidden,
            num_experts=n_experts,
            k=moe_k,
            expert_hidden_dim=expert_hidden_dim,
        )

        total_params = sum([p.numel() for p in moe.parameters()])
        print(f'moe total_params = {total_params}')

        n_expert_params = sum([p.numel() for p in moe.experts[0].parameters()])
        n_active_params = n_expert_params * moe_k + sum([p.numel() for p in moe.gate.parameters()])

        print(f'moe n_active_params = {n_active_params} ({n_active_params / total_params:.2f})')


        moe = torch.compile(moe)
        x_in = torch.randn(self.B, self.T, self.d_hidden)
        print(moe(x_in).shape)

if __name__ == '__main__':
    unittest.main()