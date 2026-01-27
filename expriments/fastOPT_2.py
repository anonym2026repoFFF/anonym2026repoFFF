import torch
import torch.nn as nn
from transformers import OPTForCausalLM, OPTConfig
from new_fff_2 import FFF
from old_fff import oldFFF
from moe import MoELayer
from ff import FF
import math
from new_moe import DenseGroupedTop1MoE

class FastOPT(OPTForCausalLM):
    def __init__(self, config, depth=3, strategy="new_FFF", moe_n_experts=4, moe_k=2):
        super().__init__(config)

        n_param = 0
        if strategy == "FF":
            print("[FastOPT] Using original FF layers with re-initialization.")
            for layer in self.model.decoder.layers:
                hidden_size = layer.fc2.out_features
                intermediate_size = layer.fc1.out_features
            
                new_fc1 = FF(layer.fc1.in_features, layer.fc1.out_features , bias=config.enable_bias if layer.fc1.bias is not None else False)
                new_fc2 = FF(layer.fc2.in_features, layer.fc2.out_features, bias=config.enable_bias if layer.fc2.bias is not None else False)


                # Replace fc1 and fc2 chain with wrapper
                layer.fc1 = new_fc1
                layer.fc2 = new_fc2

                n_param += sum(p.numel() for p in new_fc1.parameters()) + sum(p.numel() for p in new_fc1.parameters())

        elif strategy == "moe":
            print(f"[FastOPT] Replacing FFN layers with MoE layers")
            for layer in self.model.decoder.layers:
                hidden_size = layer.fc2.out_features
                intermediate_size = layer.fc1.out_features

                expert_hidden_dim = intermediate_size // moe_n_experts

                moe = MoELayer(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    num_experts=moe_n_experts,
                    k=moe_k,
                    expert_hidden_dim=expert_hidden_dim,
                )


                # Replace fc1 and fc2 chain with wrapper
                layer.fc1 = nn.Identity()  # disables fc1
                layer.activation_fn = nn.Identity()  # disables activation
                layer.fc2 = moe

                n_param += sum(p.numel() for p in moe.parameters())

        elif strategy == "old_FFF":
            print(f"[FastOPT] Replacing FFN layers with old FFF structure (depth={depth})")
            for layer in self.model.decoder.layers:
                hidden_size = layer.fc2.out_features
                intermediate_size = layer.fc1.out_features

                # make leaf_width such that all leafs together have the same width as intermediate
                # layer of original FF version
                leaf_width = intermediate_size // (2 ** depth)

                old_fff = oldFFF(
                    input_width=hidden_size,
                    leaf_width=leaf_width,
                    output_width=hidden_size,
                    depth=depth,
                )


                # Replace fc1 and fc2 chain with wrapper
                layer.fc1 = nn.Identity()  # disables fc1
                layer.activation_fn = nn.Identity()  # disables activation
                layer.fc2 = old_fff

                n_param += sum(p.numel() for p in old_fff.parameters())

        elif strategy == "new_FFF":
            print(f"[FastOPT] Replacing FFN layers with new FFF structure (depth={depth})")

            for layer in self.model.decoder.layers:
                hidden_size = layer.fc2.out_features
                intermediate_size = layer.fc1.out_features

                # print(f"Replacing layer with FFF: hidden_size={hidden_size}, intermediate_size={intermediate_size}")

                n_nodes = 2 ** (depth + 1) - 1
                parallel_size = intermediate_size // n_nodes
            
                fff = FFF(
                    input_width=hidden_size,
                    output_width=hidden_size,
                    depth=depth,
                    parallel_size=parallel_size,
                    activation=nn.GELU
                )

                # Replace fc1 and fc2 chain with wrapper
                layer.fc1 = nn.Identity()  # disables fc1
                layer.activation_fn = nn.Identity()  # disables activation
                layer.fc2 = fff

                n_param += sum(p.numel() for p in fff.parameters())

        elif strategy == "new_moe":

            for layer in self.model.decoder.layers:
                hidden_size = layer.fc2.out_features
                intermediate_size = layer.fc1.out_features

                # print(f"Replacing layer with FFF: hidden_size={hidden_size}, intermediate_size={intermediate_size}")

                n_nodes = 2 ** (depth + 1) - 1
                parallel_size = intermediate_size // n_nodes

                moe = DenseGroupedTop1MoE(
                    d_model=hidden_size,  
                    d_ff = depth,
                    parallel_size=parallel_size,
                    expert_hidden_dim = depth
                )
        
                # Replace fc1 and fc2 chain with wrapper
                layer.fc1 = nn.Identity()  # disables fc1
                layer.activation_fn = nn.Identity()  # disables activation
                layer.fc2 = moe
                n_param += sum(p.numel() for p in moe.parameters())


        else:
            raise ValueError(f"[FastOPT] Unknown strategy: {strategy}. Must be one of ['FF', 'new_FFF', 'old_FFF'].")
        

        print(f'[FastOPT] Parameters per layer: {n_param}')