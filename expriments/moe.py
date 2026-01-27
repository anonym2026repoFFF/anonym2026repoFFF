# implementation copied from: https://apxml.com/posts/how-to-implement-moe-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGate(nn.Module):
    """Gate module to select top k experts."""

    def __init__(self, input_dim, num_experts, k=1):
        super().__init__()
        self.k = k
        # Linear layer to compute logits for experts
        self.gate_linear = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        # x shape: [batch_size * seq_len, input_dim]
        # logits shape: [batch_size * seq_len, num_experts]
        logits = self.gate_linear(x)

        # Select top-k experts
        # top_k_logits shape: [batch_size * seq_len, k]
        # top_k_indices shape: [batch_size * seq_len, k]
        top_k_logits, top_k_indices = torch.topk(
            logits, self.k, dim=-1
        )

        # Apply softmax to top-k logits for weights
        # top_k_weights shape: [batch_size * seq_len, k]
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Create a sparse weight matrix for combining outputs
        # full_weights shape: [batch_size * seq_len, num_experts]
        full_weights = torch.zeros_like(logits)
        full_weights.scatter_(1, top_k_indices, top_k_weights)

        return full_weights, top_k_indices  # Return weights and indices


class Expert(nn.Module):
    """A simple feed-forward expert network."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU() # Or GeLU, SiLU etc.

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MoELayer(nn.Module):
    """Mixture of Experts layer."""

    def __init__(
            self, input_dim, output_dim, num_experts, k=1,
            expert_hidden_dim=None
            ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.output_dim = output_dim

        if expert_hidden_dim is None:
            expert_hidden_dim = input_dim * 4  # Common practice

        self.gate = TopKGate(input_dim, num_experts, k)
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden_dim, output_dim)
             for _ in range(num_experts)]
        )

    def forward(self, x):
        #print(f'start         : {x.shape}')
        # Assume x shape: [batch_size, seq_len, input_dim]
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])  # Flatten to [N, input_dim] where N = batch*seq_len

        # Get gating weights and expert indices
        # gate_weights: [N, num_experts], top_k_indices: [N, k]
        gate_weights, top_k_indices = self.gate(x)

        # Initialize final output tensor
        final_output = torch.zeros(x.shape[0], self.output_dim,
                                   device=x.device, dtype=x.dtype)

        # Get indices for batch processing
        # flat_top_k_indices: [N * k]
        flat_top_k_indices = top_k_indices.view(-1)

        # Map tokens to their assigned experts
        # Create a flat tensor of inputs for batching across experts
        # flat_x: [N * k, input_dim]
        flat_x = x.repeat_interleave(self.k, dim=0)

        # Dispatch tokens to experts and compute outputs
        expert_outputs = []
        for i in range(self.num_experts):
            # Find indices of tokens assigned to expert i
            # idx: [num_tokens_for_expert_i]
            idx = torch.where(flat_top_k_indices == i)[0]

            if idx.numel() > 0:
                # Process tokens assigned to this expert
                expert_input = flat_x[idx]
                expert_output = self.experts[i](expert_input)

                # Store output and original indices
                expert_outputs.append((idx, expert_output))

        # Combine expert outputs using gating weights
        # We need to map the results back to the original token positions
        flat_gate_weights = gate_weights.view(-1, 1)  # [N * num_experts, 1]

        for idx, output in expert_outputs:
            # Find the corresponding weights for these outputs
            # Need original token indices and expert indices
            original_indices = idx // self.k  # Get original token index (0 to N-1)
            expert_indices = flat_top_k_indices[idx]  # Which expert (0 to num_experts-1)

            # Gather the weights using original and expert indices
            weights = gate_weights[original_indices, expert_indices].unsqueeze(1)

            # Weight the expert output
            weighted_output = output * weights

            # Add to the final output tensor at the correct positions
            # Use index_add_ for scatter-add operation
            final_output.index_add_(0, original_indices, weighted_output)

        # Reshape back to original shape [batch_size, seq_len, output_dim]
        #final_output = final_output.view(original_shape[0], original_shape[1], self.output_dim)
        final_output_ret = final_output.view(*original_shape)
        #return final_output, gate_weights  # Return output and weights for aux loss

        #print(f'before reshape: {final_output.shape}')
        #print(f'after  reshape: {final_output_ret.shape}')

        return final_output_ret  # don't return gate weights for now
