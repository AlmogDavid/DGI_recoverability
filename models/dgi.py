import torch
import torch.nn as nn
from layers import AvgReadout, Discriminator
from layers.gaussian_kernel import GaussianKernel
from torch_geometric.nn import GCNConv
import torch.nn


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, loss_type):
        super(DGI, self).__init__()
        self.gcn = GCNConv(n_in, n_h, )
        self.activation = {"prelu": torch.nn.PReLU()}[activation]
        self.loss_type = loss_type
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.kernel = GaussianKernel()

        if self.loss_type == "random":
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, seq1, seq2, adj):
        h_1 = self.activation(self.gcn(seq1, adj))
        h_2 = self.activation(self.gcn(seq2, adj))

        if self.loss_type == "info_max":
            h_1 = torch.unsqueeze(h_1, 0)
            h_2 = torch.unsqueeze(h_2, 0)
            c = self.read(h_1, None)
            c = self.sigm(c)
            ret = self.disc(c, h_1, h_2)
        elif self.loss_type == "recoverability":
            d = self.kernel.compute_d(x=h_1, y=h_2)
            ret = -d  # We want to maximize d
        else:
            raise RuntimeError("Invalid loss")

        return ret

    # Detach the return variables
    def embed(self, seq, adj):
        h_1 = self.activation(self.gcn(seq, adj))
        return h_1.detach()

