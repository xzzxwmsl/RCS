import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, attrs_num, attrs_dim, hidden_dim, user_emb_dim):
        self.attrs_num = attrs_num
        self.attrs_dim = attrs_dim
        super(Generator, self).__init__()
        self.G_attr_matrix = nn.Embedding(2*attrs_num, attrs_dim)  #
        self.G_wb1 = nn.Linear(attrs_num*attrs_dim, hidden_dim)
        self.G_wb2 = nn.Linear(hidden_dim, hidden_dim)
        self.G_wb3 = nn.Linear(hidden_dim, user_emb_dim)

    def forward(self, attr_id):
        attr_present = self.G_attr_matrix(
            torch.LongTensor(attr_id.numpy()).view(-1, 18))
        attr_feature = attr_present.view(-1, self.attrs_num*self.attrs_dim)

        l1_outputs = torch.sigmoid(self.G_wb1(attr_feature))
        l2_outputs = torch.sigmoid(self.G_wb2(l1_outputs))
        fake_user = torch.sigmoid(self.G_wb3(l2_outputs))

        return fake_user
