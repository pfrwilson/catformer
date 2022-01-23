from typing import Tuple, Optional
import numpy as np
from numpy import ndarray
from torch import Tensor
import einops
from functools import cache


class AttentionGrabber():
    """
    A class for computing quantifying the attention flow between layers, tokens, and heads of a 
    transformer model. Uses techniques inspired by the "Attention Rollout" method, described by 
    Samira Abnar and Willem Zuidema in the 2020 paper found at https://arxiv.org/abs/2005.00928.
    """
    
    def __init__(self, attention_output: Tuple[Tensor, ...]):
        """[summary]

        Args:
            attention_output (Tuple[Tensor, ...]): The attention output of a transformer model, 
            stored as a tuple of nd arrays with entry i containing the attention at layer i. The 
            dimensionality of the arrays is b, h, n + 1, n + 1, where:
                b - batch size
                h - number of attention heads
                n - sequence length of model
        """
        
        self.attention_output = attention_output
        
        if isinstance(self.attention_output[0], Tensor):
            self.attention_output = list(map(
                lambda tensor : tensor.detach().numpy(), 
                attention_output
            ))
        
        b, h, n, _= self.attention_output[0].shape
        self.num_layers = len(self.attention_output)
        self.batch_size = b
        self.num_heads = h
        self.sequence_length = n
        self.true_attentions = list(map(
            AttentionGrabber.adjust_for_residual, 
            self.attention_output
        ))
        
    @cache
    def attention_rollout(self, layer: Optional[int]=None, head: Optional[int]=None):
        """Computes the attention rollout at the specified layer

        Args:
            layer (int, optional): the layer of the specified query token. Defaults to last layer.
            head (int, optional): If specified, grabs the attention from a specified head of the query token. 
                If unspecified, takes the average over the heads. 
        """    
        if layer is None:
            layer = self.num_layers - 1 
            
        if layer not in range(self.num_layers):
            raise IndexError(f'attempted to access layer {layer}, but there are only '
                             f'{self.num_layers} attention layers')
        
        if head and head not in range(self.num_heads):
            raise IndexError(f'attempted to access head {head}, but there are only '
                             f'{self.num_heads} heads.')
        
        # all heads and query tokens, attention to previous layer
        attentions = self.true_attentions[layer]
        
        # extract specific head or average
        if not head:
            attentions = einops.reduce(attentions, 'b h n1 n2 -> b n1 n2', 'mean')
        else:
            attentions = attentions[:, head, :, :]

        if layer == 0: 
            attention_rollout = attentions
        else: 
            prev_layer_attention_rollout = self.attention_rollout(layer-1, head=None)
            attention_rollout = np.einsum('bik, bkj -> bij', attentions, prev_layer_attention_rollout)
            
        return attention_rollout
            
    def get_attention_sequence(self, layer: Optional[int]=None, head: Optional[int]=None, token: Optional[int]=None):
        
        attention_rollout = self.attention_rollout(layer, head)
        
        if not token: 
            token = 0  # class token
            
        if token not in range(self.sequence_length):
            raise IndexError(f'token {token} out of range for sequence length {self.sequence_length}')
        
        return attention_rollout[:, token, :]

    def get_attention_maps(self, layer: Optional[int]=None, 
                           head: Optional[int]=None, 
                           token: Optional[int]=None,):
        """When using this class for a vision transformer, returns attentions as attention maps over an input grid
            (presumed to be square)

        Args:
            layer (Optional[int], optional): Layer at which to extract attention. Defaults to last layer.
            head (Optional[int], optional): Attention head at which to grab attention. Default behavior is 
            to average over the attention heads.
            token (Optional[int], optional): Token to query. Defaults to the class token.

        Returns:
            [type]: [description]
        """
        
        sequence_attention = self.get_attention_sequence(layer, head, token)
        sequence_attention = sequence_attention[:, 1:]       # ignore attention to class token
        
        class_attention = sequence_attention[:, 0]
        
        grid_shape = int((self.sequence_length - 1)**.5)        # assumes a square grid
        
        attention_maps = einops.rearrange(
            sequence_attention, 'b (n1 n2) -> b n1 n2', n1=grid_shape, n2=grid_shape
        )
        
        return attention_maps
        
    @staticmethod
    def adjust_for_residual(attention_matrix: ndarray) -> ndarray:
        """Converts raw attention outputs to true attention weights by averaging with the identity matrix. 
            this accounts for the effects of the residual connections, as discussed in the original paper.

        Args:
            attention_matrix (ndarray): raw attention weights
        """
        b, h, n, _ = attention_matrix.shape
        identity = np.eye(n)
        identity = einops.repeat(identity, 'n1 n2 -> b h n1 n2', b=b, h=h)
        return (identity + attention_matrix)/2
    
    
