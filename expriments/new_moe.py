import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGroupedTop1MoE(nn.Module):
    """
    Dense (fully-parallel) Grouped Top-1 MoE.

    - Input/output dim: d_model (same)
    - num_groups = parallel_size
    - group_size = 2**d_ff
    - For each token: pick TOP-1 expert inside each group, then combine the G selected outputs.
    - No Python loops over experts, no ModuleList. All experts are stored in big parameter tensors.
    - Works with batch_size > 1 and sequence length: input can be [B, T, d_model] or any [..., d_model].
    """

    def __init__(
        self,
        d_model: int,
        parallel_size: int,        # number of groups (G)
        d_ff: int,                 # exponent => group_size S = 2**d_ff
        expert_hidden_dim: int | None = None,  # hidden dim inside each expert FFN
        softmax_over_groups: bool = True,      # how to mix the one-per-group outputs
    ):
        super().__init__()
        self.d_model = d_model
        self.G = parallel_size
        self.S = 2 ** d_ff
        self.E = self.G * self.S
        self.softmax_over_groups = softmax_over_groups

        if expert_hidden_dim is None:
            expert_hidden_dim = 4 * d_model
        self.H = expert_hidden_dim

        # Gate: token -> logits over all experts
        self.gate = nn.Linear(d_model, self.E)

        # Expert parameters (per-expert FFN):
        # First layer:  x @ W1[e] + b1[e]  where W1[e]: [d_model, H]
        # Second layer: h @ W2[e] + b2[e]  where W2[e]: [H, d_model]
        self.W1 = nn.Parameter(torch.empty(self.E, d_model, self.H))
        self.b1 = nn.Parameter(torch.zeros(self.E, self.H))
        self.W2 = nn.Parameter(torch.empty(self.E, self.H, d_model))
        self.b2 = nn.Parameter(torch.zeros(self.E, d_model))

        nn.init.xavier_uniform_(self.W1)
        nn.init.zeros_(self.b1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., d_model]
        return: same shape as x
        """
        assert x.shape[-1] == self.d_model, f"Expected last dim {self.d_model}, got {x.shape[-1]}"
        orig_shape = x.shape

        # Flatten all tokens across batch/sequence/...
        x_flat = x.reshape(-1, self.d_model)  # [N, d]
        N = x_flat.size(0)

        # ---- Gating: logits per group ----
        logits = self.gate(x_flat).view(N, self.G, self.S)  # [N, G, S]
        top_vals, top_idx = logits.max(dim=-1)              # [N, G], [N, G]

        # Group mixing weights (across groups)
        if self.softmax_over_groups:
            group_w = F.softmax(top_vals, dim=-1)           # [N, G]
        else:
            group_w = torch.full_like(top_vals, 1.0 / self.G)

        # One-hot select within each group
        onehot = F.one_hot(top_idx, num_classes=self.S).to(dtype=x_flat.dtype)  # [N, G, S]

        # ---- Dense expert compute for all experts ----
        # Expand x to [N, E, d] (broadcasted, no new memory for data itself)
        x_all = x_flat.unsqueeze(1).expand(N, self.E, self.d_model)  # [N, E, d]

        # First layer for all experts: [N, E, H]
        h = torch.einsum("ned,edh->neh", x_all, self.W1) + self.b1.unsqueeze(0)
        h = F.gelu(h)

        # Second layer for all experts: [N, E, d]
        y_all = torch.einsum("neh,ehd->ned", h, self.W2) + self.b2.unsqueeze(0)

        # Reshape experts into groups: [N, G, S, d]
        y_all = y_all.view(N, self.G, self.S, self.d_model)

        # Select top-1 expert per group: [N, G, d]
        y_sel = (onehot.unsqueeze(-1) * y_all).sum(dim=2)

        # Combine groups: [N, d]
        out = (group_w.unsqueeze(-1) * y_sel).sum(dim=1)

        # Restore original shape
        return out.view(orig_shape)
