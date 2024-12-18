
### Positional Encoding

Influenced by the tensorflow lipreading project of https://github.com/afourast/deep_lip_reading

```
def sinusoid_encoding(inputs,
                      num_units,
                      zero_pad=True,
                      scale=True,
                      scope="positional_encoding",
                      T = None,
                      reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be
       constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units
      (check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be
         'num_units'
    '''

    # N, _ = inputs.get_shape().as_list()[:2]
    N = tf.shape(inputs)[0]
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)
        lookup_table = tf.cast(lookup_table, tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs
```
Rewritten to python
```

 def sinusoid_encoding(inputs, num_units, zero_pad=True, scale=True):
    """
    Sinusoidal Positional Encoding

    Args:
        inputs: Tensor of shape (batch_size, seq_len, embed_dim).
        num_units: Output dimensionality (usually matches embed_dim).
        zero_pad: Boolean. If True, positions start from 1 with 0-padding for position 0.
        scale: Boolean. If True, scales the outputs by sqrt of num_units.

    Returns:
        A tensor with the same shape as inputs, with positional encodings added.
    """
    _, seq_len, embed_dim = inputs.size()
    assert embed_dim == num_units, "num_units must match the embedding dimension of inputs."

    position = torch.arange(seq_len, dtype=torch.float32, device=inputs.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, num_units, 2, dtype=torch.float32, device=inputs.device) *
                         -(np.log(10000.0) / num_units))

    # Compute positional encoding
    pe = torch.zeros(seq_len, num_units, device=inputs.device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension
    pe = pe.unsqueeze(0)  # Shape: (1, seq_len, num_units)
    if scale:
        pe = pe * (num_units ** 0.5)

    if zero_pad:
        pe[:, 0, :] = 0  # Zero out the first position if zero_pad is True

    return inputs + pe  # Add positional encoding to inputs

 ```
### Modality encoding

Influenced by https://github.com/MASILab/lmsignatures/blob/master/TDEncoder.py

```
class ModalityEncoding(nn.Module):
    def __init__(self, dim, num_modalities=1) -> None:
        super().__init__()
        self.mes = nn.ParameterList([nn.Parameter(torch.randn(1, dim), requires_grad=True) for i in range(num_modalities)])
        # self.dim = dim

    def forward(self, ms):
        """ms: list of inputs split by modality"""
        mes = [self.mes[i].expand(*m.shape) for i, m in enumerate(ms)] # expand each modality encoding to shape of that modality
        return [ms[i] + mes[i] for i in range(len(ms))] # return list of inputs + modality encoding
        
    def encoding(self, ms):
        return [self.mes[i].expand(*m.shape) for i, m in enumerate(ms)]
```