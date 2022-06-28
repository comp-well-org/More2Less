import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import TransformerEncoder


class LSTM_early_fusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes = 2, num_layers = 2, dropout_rate=0.6):
        super(LSTM_early_fusion, self).__init__()
        self.input_dim = input_dim;
        self.hidden_dim = hidden_dim;
        self.dropout_rate = dropout_rate;
        self.num_layers = num_layers;  # number of recurrent layers, here 2 means stacking two LSTMs.
        self.num_classes = num_classes;

        # defining modules.
        # 
        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim,  num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)


    def forward(self, a, b):
        batch_size = a.size(0)
        x = torch.cat((a,b), axis=2)
        output, (h_n, c_n) = self.lstm(x)
        logits = self.fc(output[:, -1, :])
        return logits, output[:, -1, :] 



class Backbone_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, num_classes=2, num_layers=2, dropout_rate=0.5, is_training=True):
        super(Backbone_LSTM, self).__init__()
        self.input_dim = input_dim;
        self.hidden_dim = hidden_dim;
        self.num_classes = num_classes;
        self.num_layers = num_layers;
        self.dropout_rate = dropout_rate;
        self.is_training = is_training;

        # define modules.
        self.lstm = nn.LSTM(input_size= self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        
        self.fc1 = nn.Linear(2 * self.hidden_dim, 256)
        self.fc = nn.Linear(256, self.num_classes)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.bn1 = nn.BatchNorm1d(2 * self.hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        output, (h_n, c_n) = self.lstm(x)
        fc1 = self.fc1(self.dropout_layer(self.bn1(output[:, -1, :])))
        logits = self.fc(self.relu(self.dropout_layer(fc1)))
        return logits, output[:, -1, :]

class MyModel(nn.Module):
    def __init__(self, input_dim_m1, input_dim_m2, input_len, hidden_dim=64, num_classes=2, num_layers=3, dropout_rate=0.5, is_training=True):
        super(MyModel, self).__init__()
        self.input_dim_m1 = input_dim_m1;
        self.input_dim_m2 = input_dim_m2;
        self.input_len = input_len;
        self.hidden_dim = hidden_dim;
        self.num_layers = num_layers;
        self.num_classes = num_classes;
        self.drouput = dropout_rate;
        self.is_training = is_training;

        # define the backbone.
        self.backbone1 = Backbone_LSTM(self.input_dim_m1, self.hidden_dim, num_classes=self.num_classes)
        self.backbone2 = Backbone_LSTM(self.input_dim_m2, self.hidden_dim, num_classes=self.num_classes)
        #self.backbone1 = Backbone_Transformer(self.input_dim_m1,self.input_len, self.hidden_dim, num_classes=self.num_classes)
        #self.backbone2 = Backbone_Transformer(self.input_dim_m2,self.input_len, self.hidden_dim, num_classes=self.num_classes)
        #self.project_m1 = nn.Linear(self.input_dim_m1, self.hidden_dim)
        #self.project_m2 = nn.Linear(self.input_dim_m2, self.hidden_dim)

    def forward(self, m1, m2):
        #m1 = self.project_m1(m1)
        #m2 = self.project_m2(m2)
        logits_1, feats_1 = self.backbone1(m1)
        logits_2, feats_2 = self.backbone2(m2)
        return logits_1, feats_1, logits_2, feats_2