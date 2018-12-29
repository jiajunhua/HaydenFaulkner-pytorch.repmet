"""
The DML encoder / embedder
"""
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, softmax_final=False, norm_final=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1
        self.softmax_final = softmax_final
        self.norm_final = norm_final
        self.hidden_layers = nn.ModuleList([])
        assert self.num_layers >= 1
        for l in range(self.num_layers):
            if l == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_sizes[l-1]

            if l == self.num_layers-1:
                layer_output_size = output_size
            else:
                layer_output_size = hidden_sizes[l]

            self.hidden_layers.append(nn.Linear(layer_input_size, layer_output_size))

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.hidden_layers[l](x)

            if l == self.num_layers-1:
                if self.softmax_final:
                    x = F.log_softmax(x)
                elif self.norm_final:
                    x = F.normalize(x)
            else:
                x = F.relu(x)
        return x

    def extra_repr(self):
        return '{}, {}, {}, softmax_final={}, norm_final={}'.format(
            self.input_size, self.hidden_sizes, self.output_size, self.softmax_final, self.norm_final
        )

