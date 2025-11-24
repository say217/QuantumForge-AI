# model.py
import torch
import torch.nn as nn

class EnhancedGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.n_features
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_size, input_size, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.gru = nn.GRU(
            input_size=input_size*2,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn = nn.Linear(256, 256)
        self.fc = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.15)
        self.ln = nn.LayerNorm(256)

    def forward(self, x):
        conv_out1 = self.act(self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1))
        conv_out2 = self.act(self.conv2(x.permute(0, 2, 1)).permute(0, 2, 1))
        conv_out = torch.cat([conv_out1, conv_out2], dim=2)
        gru_out, _ = self.gru(conv_out)
        scores = torch.matmul(self.attn(gru_out), gru_out.transpose(1,2)) / (gru_out.size(-1)**0.5)
        weights = torch.softmax(scores.mean(dim=1), dim=1).unsqueeze(-1)
        context = (weights * gru_out).sum(dim=1)
        context = self.ln(context)
        context = self.dropout(context)
        out = self.fc(context)
        return out.squeeze(-1)
    
    
    