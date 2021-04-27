import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, attrs_num, attrs_dim, hidden_dim, user_emb_dim):
        self.attrs_num = attrs_num
        self.attrs_dim = attrs_dim
        super(Discriminator, self).__init__()
        self.D_attr_matrix = nn.Embedding(2*attrs_num, attrs_dim)  #
        self.D_wb1 = nn.Linear(attrs_num*attrs_dim + user_emb_dim, hidden_dim)
        self.D_wb2 = nn.Linear(hidden_dim, hidden_dim)
        self.D_wb3 = nn.Linear(hidden_dim, user_emb_dim)

    def forward(self, attr_id, user_emb):
        attr_present = self.D_attr_matrix(
            torch.LongTensor(attr_id.numpy()).view(-1, 18))
        attr_feature = attr_present.view(-1, self.attrs_num*self.attrs_dim)
        emb = torch.cat([attr_feature, user_emb], 1)

        l1_outputs = torch.sigmoid(self.D_wb1(emb))
        l2_outputs = torch.sigmoid(self.D_wb2(l1_outputs))
        D_logit = self.D_wb3(l2_outputs)
        D_prob = torch.sigmoid(D_logit)

        return D_prob, D_logit
