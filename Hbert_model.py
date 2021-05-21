import torch.nn as nn
import torch
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Hbert(nn.Module):
    def __init__(
            self,
            model_name,
            n_classes,
            hidden_size,
            n_layers,
            dropout_p,
    ):
        self.model_name = model_name
        self.n_class = n_classes
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.bertmodel = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.gru = nn.GRU(
            input_size=self.bertmodel.config.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, self.n_class),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x):
        # |x| = (batch_size, length)
        # |output| = (batch_size, Sequence_length, hidden_states)
        # last_hidden_state = (batch_size, Sequence_length, features)
        output = self.bertmodel(x)
        last_hidden_state=output[2][24]

        out,_ = self.gru(last_hidden_state)
        out_forward = out[:, -1, :self.hidden_size]
        out_backward = out[:, 0, self.hidden_size:]
        pre_classifier = torch.cat((out_forward, out_backward), 1)
        y_hat = self.classifier(pre_classifier)

        return y_hat

class Bert_GRU_TOKEN_MIX_GRU(nn.Module):
    def __init__(
            self,
            model_name,
            n_classes,
            hidden_size,
            n_layers,
            dropout_p,
    ):
        self.model_name = model_name
        self.n_class = n_classes
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.bertmodel = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.gru = nn.GRU(
            input_size=self.bertmodel.config.hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.n_class),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x):

        output = self.bertmodel(x)
        embeddings = output

        pooled_output1 = embeddings[2][12]
        pooled_output2 = embeddings[2][24]
        pooled_output = torch.cat((pooled_output1, pooled_output2), 2)
        pooled_output = pooled_output.permute(1, 0, 2)
        out, _ = self.gru(pooled_output)

        x = torch.cat((out[:, 0, :self.hidden_size], out[:, -1, self.hidden_size:]), 1)
        x = self.classifier(x)


        return x

