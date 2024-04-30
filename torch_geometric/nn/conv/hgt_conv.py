import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax, is_sparse, is_torch_sparse_tensor
from torch_geometric.utils.hetero import construct_bipartite_edge_index

class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        edge_attribute_dims: Optional[Dict[EdgeType, int]] = None,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = set([key[-1] for key in self.edge_types])

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)
        attr_dim = 0
        if edge_attribute_dims is not None:
            attr_dim = max(edge_attribute_dims.values())
        self.k_rel = HeteroLinear(dim + attr_dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim + attr_dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads
        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        edge_attrs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])
        print(f"ks dim: {[k.shape for k in ks]}")
        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        print(f"ks dim: {ks.shape}")
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        print(f"type list: {[t.shape for t in type_list]}")
        print(f"type list: {type_list}")
        type_vec = torch.cat(type_list, dim=1).transpose(0,1)
        k = ks.view(H, -1, D).transpose(0, 1)
        v = vs.view(H, -1, D).transpose(0, 1)
        print(f"type_vec: {type_vec.shape}")
        return k, v, type_vec, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],  # Support both.
        edge_weight_dict: Optional[Dict[EdgeType, Tensor]] = None,
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[torch.Tensor]]` - The output node
            embeddings for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        print(f"edge index shape: {edge_index_dict}")
        if edge_weight_dict is None:
            edge_weight_dict = self.p_rel
        if edge_attr_dict is None:
            edge_attr_dict = {}
            for k, v in edge_index_dict.items():
                if is_sparse(v):
                    if is_torch_sparse_tensor(v):
                        edge_attr_dict[k] = torch.zeros((len(v.values()), 0))
                    else:
                        print(f"v: {v.storage.value()}, {v.storage._col}")
                        edge_attr_dict[k] = torch.zeros((v.numel(), 0))
                else:
                    edge_attr_dict[k] = torch.zeros((v.shape[1], 0))
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, type_vec, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict, )
        print(f"type_vec shape @ forward: {type_vec.shape}")
        edge_index, edge_weight, edge_attr = construct_bipartite_edge_index(
            edge_index_dict,
            src_offset,
            dst_offset,
            edge_attr_dict = edge_attr_dict,
            edge_weight_dict = edge_weight_dict,
            num_nodes=k.size(0))
        print(f"edge_index @forward: {edge_index}")
        print(f"edge_attr @forward: {edge_attr}")
        print(f"edge_weight @forward: {edge_weight}")
        #k.shape == q.shape == v.shape == (n_total_nodes, H, D)
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_weight=edge_weight, edge_attr=edge_attr, edge_type=type_vec)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_weight: Tensor, edge_attr: Tensor, edge_type_i: Tensor, edge_type_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        print(f"q_i dimension:{q_i.shape}; k_j dimension:{k_j.shape}; edge_attr dimension:{edge_attr.shape}, edge_attr values:{edge_attr}; edge_weight dimension:{edge_weight.shape}, edge_weight values:{edge_weight};")
        n_edges, H, D = q_i.shape
        # edge_attr.shape == (n_edges, n_attr_dim)
        # Input for the k_rel should be (n_samples, n_dim) and (n_samples,)
        type_vec = edge_type_j
        print(f"type_vec @ message: {type_vec}")
        #print(f"k_j @ message: {k_j}")
        k_j = self.k_rel(torch.cat([k_j, edge_attr.unsqueeze(1).expand(-1, H, -1)], axis=-1).view(-1, D), type_vec.flatten()).view(H, -1, D).transpose(0, 1)
        print(f"edge_attr: {edge_attr}")
        v_j = self.v_rel(torch.cat([v_j, edge_attr.unsqueeze(1).expand(-1, H, -1)], axis=-1).view(-1, D), type_vec.flatten()).view(H, -1, D).transpose(0, 1)
        alpha = (q_i * k_j).sum(dim=-1)# * edge_weight #edge_attr # dimension?
        alpha = alpha / math.sqrt(q_i.size(-1))
        
        alpha = softmax(alpha, index, ptr, size_i) * edge_weight
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.reshape(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
