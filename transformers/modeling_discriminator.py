import torch
from torch import nn
from torch.nn import BCELoss

from .modeling_bert import BertLayer

LAYER_CLASSES = {
    'transformer': TransformerEncoder,
}

class Discriminator(nn.Module):
    def __init__(self, config, model_config, layer):
        super(Discriminator, self).__init__()
        discriminator = config.discriminator
        self.encoder = ENCODER_CLASSES[config.discriminator]
        self.linear = nn.Linear(model_config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.confused_layer=layer-1

    def forward(self, hidden_states, mask, labels):
        hidden_states = self.encoder(hidden_states, mask)
        self.logits = self.linear(hidden_states)
        dis_loss = self.loss_fn(self.sigmoid(self.logits), labels)
        return dis_loss

    def __repr__(self):
        return f"Discriminator for layer {self.confused_layer+1}"

    @classmethod
    def create_discriminator(cls, config, model_config):
        return nn.ModuleList([cls(config, model_config, layer) for layer in config.confused_layers])


class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        self.pooler_type = config.discriminator_pooler_type

    def forward(hidden_states, mask=None):
        if self.pooler_type='first':
            return hidden_states[:, 0]
        elif self.pooler_type='last':
            return hidden_states[:, -1]
        # TODO cnn should change mask
        elif self.pooler_type='mean':
            if mask is None:
                return torch.mean(hidden_states, dim=1)
            else:
                replace_hidden_states=hidden_states.masked_fill(mask, 0.0)
                value_sum = torch.sum(replace_hidden_states, dim=1)
                value_count = torch.sum(mask.float(), dim=1)
                assert value_count!=0
                return value_sum / value_count
        raise ValueError("Unrecognized Pooler Type")


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = Pooler(config)
    def forward(self, hidden_states, attention_mask):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
        output = self.pooler(hidden_states, mask=attention_mask)
        return output

