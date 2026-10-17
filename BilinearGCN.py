from typing import Optional
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

from jaxtyping import Float

from torch_geometric.nn import GCNConv
@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


# Bilinear transformation
class Bilinear(nn.Linear):
    """A bilinear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {"relu" :nn.ReLU(), "silu" :nn.SiLU(), "gelu" :nn.GELU(), None: nn.Identity()}[gate]

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right

    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]



class BiGCNConv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        gate: Optional[str] = "relu",
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.num_nodes = num_nodes

        self._cached_edge_index = None
        self._cached_adj_t = None

        # Interaction intensity
        self.alpha = None

        self.record_aij = False  # Set to True for explanation; keep False during training
        self.last_aij_edge_index = None  # Store the actual edge_index used in forward's propagate
        self.last_aij = None  # Store the actual edge_weight (= a_ij) used in forward's propagate
        self.last_h = None  # Optional: store h (x after LN) of this layer, helpful for explaining neighbor projections

        # self.bilin = Bilinear(in_channels, out_channels)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # self.bilinear_tr = nn.Linear(in_channels, out_channels, bias=bias)
        self.bilinear_tr = Bilinear(d_in=in_channels, d_out=out_channels, bias=bias, gate=gate)
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')



        # ===== Step2-4: Edge-wise interaction (computed ONLY on edges, O(E*d)) =====
        # Step2: two different projection matrices for two endpoints of an edge:
        #   z_ij = (W h_i) ⊙ (V h_j)
        self.edge_proj_src = nn.Linear(out_channels, out_channels, bias=False)  # W
        self.edge_proj_dst = nn.Linear(out_channels, out_channels, bias=False)  # V

        # Step3: relation projection vector r: alpha_ij = r^T z_ij
        self.rel_vec = Parameter(torch.empty(out_channels))

        # Step4: learnable scalar lambda to control enhancement strength
        # Initialize small for stability.
        self.register_buffer('lambda_struct', torch.tensor(0.2, dtype=torch.float))    

        # Optional switch: allow turning off dynamic edge weights for ablations
        self.use_dynamic_edge_weight = True




        # Bilinear interaction module: input is [N] for each channel dimension
        # self.bilinear = Bilinear(d_in=num_nodes, d_out=num_nodes, bias=False, gate=gate)
        # self.classifier = nn.Linear(in_channels, out_channels, bias=bias)

        self.linear_tr = nn.Linear(out_channels, out_channels, bias=bias)



        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        # Step1
        self.bilinear_tr.reset_parameters()

        # Step2-4
        self.edge_proj_src.reset_parameters()
        self.edge_proj_dst.reset_parameters()
        nn.init.xavier_uniform_(self.rel_vec.view(1, -1))

        if self.bias is not None:
            zeros(self.bias)

        # cache is not useful when edge weights depend on features
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            edge_weight: OptTensor = None,
            edge_mask: Optional[Tensor] = None,  # [E]
    ) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)


        # h is defined as the LN output (your definition)
        h = x

        # Step2-4: compute dynamic edge weights on edges (O(E*d))
        if self.use_dynamic_edge_weight:
            if not isinstance(edge_index, Tensor):
                raise NotImplementedError("Dynamic edge weights are implemented for Tensor edge_index.")

            base_w = torch.ones(edge_index.size(1), device=h.device, dtype=h.dtype) \
                if edge_weight is None else edge_weight

            src, dst = edge_index[0], edge_index[1]

            # Step2: z_ij = (W h_i) ⊙ (V h_j)
            hs = self.edge_proj_src(h)  # [N, out]
            ht = self.edge_proj_dst(h)  # [N, out]
            z = hs[src] * ht[dst]  # [E, out]

            # Step3: alpha_ij = r^T z_ij
            alpha_a = (z * self.rel_vec).sum(dim=-1)  # [E]
            # alpha = F.softplus(alpha)  # keep >=0 for stable GCN normalization
            self.alpha = alpha_a.masked_fill(src == dst, 0.0)  # Alpha excludes self-loops
            # Step4: fuse into adjacency weights (before gcn_norm)
            edge_weight = base_w + self.lambda_struct * self.alpha
            # ===== explanation-time edge mask (per-layer) =====
            if edge_mask is not None:
                E = edge_mask.size(0)
                E_new = edge_weight.size(0)

                if E_new > E:
                    # Self-loops have been added
                    num_loops = E_new - E

                    loop_mask = torch.ones(
                        num_loops,
                        device=edge_mask.device,
                        dtype=edge_mask.dtype
                    )

                    edge_mask = torch.cat([edge_mask, loop_mask], dim=0)

                # edge_mask: [E], 0/1 or soft, aligned with edge_index
                edge_weight = edge_weight * edge_mask

        if (not self.training) and self.record_aij:
            # edge_index/edge_weight here are EXACTLY what propagate uses
            self.last_aij_edge_index = edge_index.detach()
            self.last_aij = None if edge_weight is None else edge_weight.detach()
            self.last_h = x.detach()


        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)