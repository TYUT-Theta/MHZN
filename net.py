import torch.nn
import scipy.sparse as sp
from layer import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

class TDG4MSF(nn.Module):
    def __init__(self, gcn_true,buildA_true,num_nodes, device,  dropout=0.3, subgraph_size=20, node_dim=40, residual_channels=32, seq_length=12,  layers=1,  tanhalpha=3, layer_norm_affline=True,output_attention=False,activation="relu"):
        super(TDG4MSF, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.dropout1=torch.nn.Dropout(0.1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.mlgnn = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.output_attention = output_attention
        self.dropout1 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.linear1 = nn.Linear(seq_length, 64)
        self.linear2 = nn.Linear(16, seq_length)
        self.linear3 = nn.Linear(seq_length, 64)
        self.linear4 = nn.Linear(64, 1)
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha)
        self.seq_length = seq_length
        for i in range(layers):
            kernel_size = 25
            self.decompsition1 = series_decomp(kernel_size)
            self.decompsition2 = series_decomp(kernel_size)
            self.individual ='store_true'
            self.channels = num_nodes

            self.Linear_Seasonal = nn.Linear(seq_length, seq_length)
            self.Linear_Trend = nn.Linear(seq_length, seq_length)
            if self.gcn_true:
                self.mlgnn.append(Gated_GNNML())
            self.norm.append(
                    LayerNorm((residual_channels, num_nodes, self.seq_length), elementwise_affine=layer_norm_affline))
        self.layers = layers
        self.skip0 = nn.Conv1d(in_channels=seq_length, out_channels=seq_length, kernel_size=1,bias=True)
        self.skip1 = nn.Conv1d(in_channels=seq_length, out_channels=1, kernel_size=1,bias=True)
        self.e1 = nn.Conv1d(in_channels=seq_length, out_channels=seq_length, kernel_size=1,bias=True)
        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None):
        input = torch.squeeze(input,dim=1)
        seq_len = input.size(2)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        skip = (self.skip0(input.permute(0, 2, 1))).permute(0,2,1)
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                    adp1 = adp.transpose(1, 0)
                    adp = adp.cpu().detach().numpy()
                    adp = sp.coo_matrix(adp)
                    indices = np.vstack((adp.row, adp.col))
                    adp = torch.LongTensor(indices)
                    adp1 = adp1.cpu().detach().numpy()
                    adp1 = sp.coo_matrix(adp1)
                    indices1 = np.vstack((adp1.row, adp1.col))
                    adp1 = torch.LongTensor(indices1)  #
                else:
                    adp = self.gc(idx)
                    adp1 = adp.transpose(1, 0)
                    adp = adp.cpu().detach().numpy()
                    adp = sp.coo_matrix(adp)
                    indices = np.vstack((adp.row, adp.col))
                    adp = torch.LongTensor(indices)
                    adp1 = adp1.cpu().detach().numpy()
                    adp1 = sp.coo_matrix(adp1)
                    indices1 = np.vstack((adp1.row, adp1.col))
                    adp1 = torch.LongTensor(indices1)

        for i in range(self.layers):
            residual = input
            x = input.permute(0, 2, 1)
            seasonal, trend_1 = self.decompsition1(x)
            seasonal, trend_1 = seasonal.permute(0, 2, 1), trend_1.permute(0, 2, 1)
            seasonal, trend_2 = self.decompsition2(seasonal.permute(0, 2, 1))
            seasonal = seasonal.permute(0, 2, 1)
            seasonal_output = self.Linear_Seasonal(seasonal)
            trend_output = self.Linear_Trend(trend_1 + trend_2.permute(0, 2, 1))
            x = seasonal_output + trend_output
            s = x
            s = (self.skip1(s.permute(0,2,1))).permute(0,2,1)
            skip = s + skip
            if self.gcn_true:
                x = self.linear1(x)
                adp = adp.to(device)
                adp1 = adp1.to(device)
                x = self.mlgnn[i](x, adp)+self.mlgnn[i](x, adp1)
                x = self.linear2(x)
            else:

                x = self.residual_convs[i](x)
            x = x + residual[:, :, -x.size(2):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)
        x = (self.e1(x.permute(0,2,1))).permute(0,2,1)
        skip = x+skip
        x = F.relu(skip)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
