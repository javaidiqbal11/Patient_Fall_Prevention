import torch.nn as nn

# More deep learning models can be trained for the best results in the detection

class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, h_RNN_layers=2, h_RNN=256, drop_p=0.2, num_classes=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.h_RNN_layers = h_RNN_layers
        self.h_RNN = h_RNN
        self.drop_p = drop_p
        if h_RNN_layers < 2:
            drop_p = 0
        self.num_classes = num_classes
        self.LSTM = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            dropout=drop_p,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.h_RNN, self.num_classes)

    def forward(self, x, h_s=None):
        self.LSTM.flatten_parameters()
        RNN_out, h_s = self.LSTM(x, h_s)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        out = self.fc1(RNN_out[:, -1, :])
        return out, h_s
