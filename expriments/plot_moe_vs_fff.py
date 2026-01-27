from itertools import chain

import numpy as np
import torch
import matplotlib.pyplot as plt

from moe import MoELayer
from new_fff import FFF


def moe_params(d_hidden, d_intermediate, n_experts, moe_k):
    expert_hidden_dim = d_intermediate // n_experts

    moe = MoELayer(
        input_dim=d_hidden,
        output_dim=d_hidden,
        num_experts=n_experts,
        k=moe_k,
        expert_hidden_dim=expert_hidden_dim,
    )

    total_params = sum([p.numel() for p in moe.parameters()])

    n_expert_params = sum([p.numel() for p in moe.experts[0].parameters()])
    real_k = min(n_experts, moe_k)
    n_active_params = n_expert_params * real_k + sum([p.numel() for p in moe.gate.parameters()])
    frac_active_params = n_active_params / total_params
    return frac_active_params


def new_fff_params(d_hidden, d_intermediate, depth):
    n_nodes = 2 ** (depth + 1) - 1
    parallel_size = d_intermediate // n_nodes
    # print(f'depth = {depth}, n_nodes = {n_nodes}, parallel_size = {parallel_size}')

    fff_new = FFF(
        input_width=d_hidden,
        output_width=d_hidden,
        depth=depth,
        parallel_size=parallel_size,
        activation=torch.nn.GELU,
    )

    total_params = sum([p.numel() for p in fff_new.parameters()])

    n_active_nodes = (fff_new.depth + 1) * fff_new.parallel_size
    n_active_node_params = n_active_nodes * fff_new.input_width
    n_output_proj_params = n_active_nodes * fff_new.output_width
    n_active_params = n_active_node_params + n_output_proj_params
    frac_active_params = n_active_params / total_params
    return frac_active_params


def plot():
    d_hidden = 4096 * 2
    d_intermediate = 4096
    depth = range(3, 11, 2)
    n_experts = list(range(2, 10, 2)) + list(range(10, 50, 4))
    moe_k = [1, 2, 4, 8]

    frac_used_fff = [new_fff_params(d_hidden, d_intermediate, d) for d in depth]
    frac_used_moe = [
        [moe_params(d_hidden, d_intermediate, n_ex, k) for n_ex in n_experts]
        for k in moe_k
    ]

    fig, ax1 = plt.subplots()

    ax1.set_ylabel('FFF Depth', color='orange')
    ax1.scatter(frac_used_fff, depth, label='FFF', c='orange', marker='o')
    for frac, d in zip(frac_used_fff, depth):
        ax1.hlines(d, 0.0, frac, colors='orange', linestyles='--', linewidth=0.5)
        ax1.vlines(frac, d, depth[-1], colors='orange', linestyles='--', linewidth=0.5)
    ax1.tick_params('y', labelcolor='orange')
    ax1.set_xlim((0.0, 1.05))

    ax2 = ax1.twinx()
    ax2.set_ylabel('MoE Experts', color='blue')
    ax2.tick_params('y', labelcolor='blue')
    ax2.set_yticks(range(n_experts[-1])[::2])
    markers = ['o', 'P', 's', 'x', 'D']
    for frac_used, k, m in zip(frac_used_moe, moe_k, markers):
        ax2.scatter(frac_used, n_experts, c='blue', marker=m, label=f'MoE (k={k})')
        if k == 1:
            for frac, n_ex in zip(frac_used, n_experts):
                ax2.hlines(n_ex, frac, 1.05, colors='blue', linestyles='--', linewidth=0.5)

    plt.legend()
    plt.show()

def avg_act_params_literature():
    # numbers from https://arxiv.org/pdf/2407.06204
    active_counts = [2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 8, 8, 16,
                     2, 2, 2, 8, 4, 2, 2, 2, 8, 9]
    total_counts = [2048, 2048, 512, 128, 128, 128, 64, 2048, 64, 64, 64, 64, 128, 128, 32, 64, 32,
                    64, 8, 8, 16, 16, 8, 64, 66, 132, 16, 32, 32, 64, 16, 16, 16, 32, 64, 257]
    d_expert = [8192, 8192, 8192, 8192, 2048, 2816, 10240, 6144, 3072, 8192, 16384, 32768, 4096,
                8192, 4096, 8192, 2816, 20480, 14336, 16384, 688, 688, 1376, 320, 1408, 1024, 3072,
                8192, 12288, 1408, 10752, 14336, 12288, 8192, 1024, 2048]
    d_model = [1024, 1024, 1024, 1024, 768, 1024, 4096, 2080, 768, 2048, 4096, 8192, 1024, 2048,
               1024, 2048, 1024, 5120, 4096, 6144, 4096, 4096, 4096, 1280, 2048, 4096, 768, 2048,
               3072, 2048, 6144, 4096, 4608, 2048, 2048, 7168]

    active_frac = []
    for k, n_ex, d_ex, d_in in zip(active_counts, total_counts, d_expert, d_model):
        active_frac.append(moe_params(d_in, d_ex * n_ex, n_ex, k))

    avg_active_experts = round(np.mean(active_counts))
    print('Active Experts:', avg_active_experts)
    avg_total_experts = round(np.mean(total_counts))
    print('Total Experts:', avg_total_experts)
    avg_active_frac = round(np.mean(active_frac))
    print('Active Fraction:', avg_active_frac)


if __name__ == '__main__':
    plot()
    #avg_act_params_literature()
