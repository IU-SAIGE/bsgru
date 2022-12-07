import math
import torch
import torch.nn

from typing import Optional, Tuple, Union
from torch.nn.utils.rnn import PackedSequence
from torch.nn.functional import one_hot
from torch import sigmoid, tanh, Tensor


def check(t, name: str = ''):
    if torch.isnan(t).any(): raise ValueError(f'{name} has nan values')
    if torch.isinf(t).any(): raise ValueError(f'{name} has inf values')
    return


def repeat_interleave(v: Tensor, num_repeats: int, dim: int = -1):
    '''Manual reimplementation of torch.repeat_interleave that creates the
    output tensor as a view of the input tensor.'''
    old_shape = list(v.shape)
    old_shape[dim] = -1
    expand_dims = [-1] * (v.ndim + 1)
    expand_dims[dim] = num_repeats
    return v.unsqueeze(dim).expand(*expand_dims).contiguous().view(*old_shape)


class BlockSparseGRU(torch.nn.GRU):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_blocks: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        beta: float = 1.,
        dropout: float = 0.,
        device=None,
        dtype=None) -> None:

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        factory_kwargs = {'device': device, 'dtype': dtype}
        if num_blocks < 1:
            raise ValueError(
                'num_blocks must be >= 1')
        if (hidden_size/num_blocks) != (hidden_size//num_blocks):
            raise ValueError(
                'hidden_size must be evenly divisible by num_blocks')
        if bidirectional:
            raise NotImplementedError('no support for bidirectional GRU')
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.beta = beta
        self.dropout = dropout
        num_directions = 2 if bidirectional else 1
        for layer in range(num_layers):
            for direction in range(num_directions):
                l_input_size = input_size
                if layer != 0:
                    l_input_size = hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''
                setattr(self, f'weight_ik_l{layer}{suffix}', torch.nn.Parameter(
                    torch.empty((num_blocks, l_input_size), **factory_kwargs)))
                setattr(self, f'weight_hk_l{layer}{suffix}', torch.nn.Parameter(
                    torch.empty((num_blocks, hidden_size), **factory_kwargs)))
                if bias:
                    setattr(self, f'bias_ik_l{layer}{suffix}', torch.nn.Parameter(
                        torch.empty(num_blocks, **factory_kwargs)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        stdv *= math.sqrt(6)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.proj_size != 0:
            s += ', proj_size={proj_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.num_blocks > 0:
            s += ', num_blocks={num_blocks}'
        return s.format(**self.__dict__)

    def forward(
        self,
        inputs: Union[torch.Tensor, PackedSequence],
        h_0: Optional[torch.Tensor] = None,
        k_0: Optional[torch.Tensor] = None):

        # figure out batch sizes and indices
        orig_input = inputs
        if isinstance(orig_input, PackedSequence):
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            is_batched = inputs.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                inputs = inputs.unsqueeze(batch_dim)
                if h_0 is not None:
                    if h_0.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, h_0 should also be 2-D "
                            f"but got {h_0.dim()}-D tensor")
                    h_0 = h_0.unsqueeze(1)
            else:
                if h_0 is not None and h_0.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, h_0 should also be 3-D "
                        f"but got {h_0.dim()}-D tensor")
            batch_sizes = None
            max_batch_size = inputs.size(batch_dim)
            sorted_indices = None
            unsorted_indices = None


        # ensures that the hidden state matches the input sequence
        num_directions = 2 if self.bidirectional else 1
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)
        else:
            h_0 = self.permute_hidden(h_0, sorted_indices)

        if k_0 is None:
            k_0 = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.num_blocks,
                             dtype=inputs.dtype, device=inputs.device)

        # run PyTorch-provided checks
        self.check_forward_args(inputs, h_0, batch_sizes)

        # loop over layers and over time and over blocks
        hiddens = []
        layer_blocks = []
        for l in range(self.num_layers):

            N = int(self.input_size if l == 0 else self.hidden_size)
            M = self.num_blocks
            L = self.block_size
            H = self.hidden_size

            # retrieve model parameters for this layer
            W_z, W_r, W_h = getattr(self, f"weight_ih_l{l}").view(3, H, N)
            U_z, U_r, U_h = getattr(self, f"weight_hh_l{l}").view(3, 1, H, H)
            W_k = getattr(self, f"weight_ik_l{l}")
            U_k = getattr(self, f"weight_hk_l{l}")
            if self.bias:
                b_z, b_r, b_h = getattr(self, f"bias_ih_l{l}").view(3, H)
                b_k = getattr(self, f"bias_ik_l{l}")
            else:
                b_z, b_r, b_h = torch.zeros(3, H)
                b_k = torch.zeros(M)


            h_prev = h_0[l, :, :]
            k_prev = k_0[l, :, :]

            outputs = []
            blocks = []

            for t in range(inputs.size(1)):

                x = inputs[:, t, :]
                check(x, 'x')

                # predict the current active block for this time step
                k_next = x.mm(W_k.T) + h_prev.mm(U_k.T) + b_k
                check(k_next, 'k_next 1')
                k_next = softmax(self.beta * k_next, dim=-1)
                check(k_next, 'k_next 2')

                # if the model is in eval mode, replace softmax with hardmax
                if not self.training:
                    k_next = one_hot(k_next.argmax(-1), num_classes=M).float()
                    k_prev = one_hot(k_prev.argmax(-1), num_classes=M).float()

                # use repeat-interleave so that k_next and k_prev can be
                # element-wise multiplied with the hidden-to-hidden weights
                k_n = repeat_interleave(k_next, self.block_size, dim=-1)
                check(k_n, 'k_n')
                k_p = repeat_interleave(k_prev, self.block_size, dim=-1)
                check(k_p, 'k_p')
                k_np = torch.matmul(k_n.unsqueeze(2), k_p.unsqueeze(1))
                check(k_np, 'k_np')
                h_p = h_prev.unsqueeze(1)
                check(h_p, 'h_p')

                # compute update gate
                z_gate = sigmoid(
                    k_n * x.mm(W_z.T)
                    + h_p.matmul(k_np*U_z).sum(1)
                    + b_z
                )
                check(z_gate, 'z_gate')

                # compute reset gate
                r_gate = sigmoid(
                    k_n * x.mm(W_r.T)
                    + h_p.matmul(k_np*U_r).sum(1)
                    + b_r
                )
                check(r_gate, 'r_gate')

                # compute new gate
                h_gate = tanh(
                    k_n * x.mm(W_h.T)
                    + (r_gate * (h_p.matmul(k_np*U_h).sum(1)))
                    + b_h
                )
                check(h_gate, 'h_gate')

                # compute output
                h_next = (
                    z_gate * h_gate # (k_n * (1 - z_gate) * h_gate)
                    + (1 - z_gate) * h_prev # + (z_gate * h_p.matmul(k_np).sum(1))
                )
                check(h_next, 'h_next')

                # use the output from this time step as the input to the next
                outputs.append(h_next)
                blocks.append(k_next)
                h_prev = h_next
                k_prev = k_next

            # use this layer's outputs as the next layer's inputs
            inputs = torch.stack(outputs, 1)

            # keep track of the final hidden state from this layer
            hiddens.append(h_next)

            # keep track of the block indices from this layer
            layer_blocks.append(torch.stack(blocks, 1))

        # use the same return signature as PyTorch's GRU
        output, hidden = inputs, torch.stack(hiddens)

        return output, hidden, torch.stack(layer_blocks)
