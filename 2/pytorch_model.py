import pandas as pd
import torch
import numpy as np
import support
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


attr_num = 18
attr_dim = 5
batch_size = 1024
hidden_dim = 100
user_emb_dim = 18
learning_rate = 0.0001
alpha = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.G_attr_matrix = nn.Embedding(2*attr_num, attr_dim)  #
        self.G_wb1 = nn.Linear(attr_num*attr_dim, hidden_dim)
        self.G_wb2 = nn.Linear(hidden_dim, hidden_dim)
        self.G_wb3 = nn.Linear(hidden_dim, user_emb_dim)

    def forward(self,attr_id):
        attr_present = self.G_attr_matrix(torch.LongTensor(attr_id.numpy()).view(-1,18)) 
        attr_feature = attr_present.view(-1, attr_num*attr_dim)
        
        l1_outputs = torch.sigmoid(self.G_wb1(attr_feature))
        l2_outputs = torch.sigmoid(self.G_wb2(l1_outputs))
        fake_user = torch.sigmoid(self.G_wb3(l2_outputs))

        return fake_user

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.D_attr_matrix = nn.Embedding(2*attr_num, attr_dim)  #
        self.D_wb1 = nn.Linear(attr_num*attr_dim + user_emb_dim, hidden_dim)
        self.D_wb2 = nn.Linear(hidden_dim, hidden_dim)
        self.D_wb3 = nn.Linear(hidden_dim, user_emb_dim)

    def forward(self, attr_id, user_emb):
        attr_present = self.D_attr_matrix(torch.LongTensor(attr_id.numpy()).view(-1,18))  
        attr_feature = attr_present.view(-1, attr_num*attr_dim)
        emb = torch.cat([attr_feature, user_emb], 1)

        l1_outputs = torch.sigmoid(self.D_wb1(emb))
        l2_outputs = torch.sigmoid(self.D_wb2(l1_outputs))
        D_logit = self.D_wb3(l2_outputs)
        D_prob = torch.sigmoid(D_logit)

        return D_prob, D_logit

def get_data(begin, end):
    train_user_batch, train_item_batch, train_attr_batch,train_user_emb_batch = support.get_traindata(begin, end)
    counter_user_batch, counter_item_batch, counter_attr_batch, counter_user_emb_batch = support.get_negdata(begin, end)

    return torch.Tensor(train_attr_batch),torch.Tensor(train_user_emb_batch),torch.Tensor(counter_attr_batch), torch.Tensor(counter_user_emb_batch)

def train():
    generator = Generator()
    discriminator = Discriminator()
    generator.to(device)
    discriminator.to(device)

    print("device:", device)
    # print("Generator:")
    # print(generator)
    # print("Discriminator:")
    # print(discriminator)

    optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=alpha)
    optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=alpha)

    G_losses = []
    D_losses = []
    BCELoss = nn.BCELoss()
    p10_save = []
    p20_save = []
    m10_save = []
    m20_save = []
    g10_save = []
    g20_save = []


    support.shuffle()
    support.shuffle2()

    num_epochs = 300
    for epoch in range(num_epochs): 
        start = time.time()
        for D_it in range(1):
            index = 0

            while index < 253236:
                # discriminator.zero_grad()

                if index + batch_size <= 253236:
                    train_attr_batch,train_user_emb_batch,counter_attr_batch, counter_user_emb_batch = get_data(index, index + batch_size)
                index = index + batch_size

                train_attr_batch.to(device)
                train_user_emb_batch.to(device)
                counter_attr_batch.to(device)
                counter_user_emb_batch.to(device)

                fake_user_emb = generator(train_attr_batch)
                fake_user_emb.to(device)

                D_real, D_logit_real = discriminator(train_attr_batch, train_user_emb_batch)
                D_fake, D_logit_fake = discriminator(train_attr_batch, fake_user_emb)
                D_counter, D_logit_counter = discriminator(counter_attr_batch, counter_user_emb_batch)
                
                D_loss_real = BCELoss(D_real, torch.ones_like(D_real)).mean()
                D_loss_fake = BCELoss(D_fake, torch.zeros_like(D_fake)).mean()
                D_loss_counter = BCELoss(D_counter, torch.zeros_like(D_counter)).mean()
                D_loss = D_loss_real + D_loss_fake + D_loss_counter
                optimizerD.zero_grad()
               
                D_loss.backward()
                optimizerD.step()
        D_losses.append(D_loss.item())

        for G_it in range(1):
            index = 0

            while index < 253236:
                if index + batch_size <= 253236:
                    train_attr_batch,_,_,_ = get_data(index, index + batch_size)
                index = index + batch_size

                train_attr_batch.to(device)

                fake_user_emb = generator(train_attr_batch)
                fake_user_emb.to(device)
                D_fake, D_logit_fake = discriminator(train_attr_batch, fake_user_emb)

                G_loss = BCELoss(D_fake, torch.ones_like(D_fake)).mean()

                optimizerG.zero_grad()
                G_loss.backward()
                optimizerG.step()

        G_losses.append(G_loss.item())

        end = time.time()
        print("epoch: {}/{}, D_loss:{:.4f}, G_loss:{:.4f}，time:{:.4f}".format(epoch, 300, D_loss, G_loss, end-start))

        if epoch % 10 == 0 :
            torch.save(generator.state_dict(), 'save_generator/epoch'+ str(epoch) + '.pt')
            torch.save(discriminator.state_dict(), 'save_discriminator/epoch'+ str(epoch) + '.pt')

        if epoch % 20 == 0 :
            print('----------------------')
            print('BEGIN TEST:')
            test_items, test_attrs = support.get_testdata()
            test_attrs = torch.Tensor(test_attrs)
            test_attrs.to(device)

            gent = Generator()
            gent.load_state_dict(torch.load('save_generator/epoch'+str(epoch)+'.pt'))
            gent.to(device)
            test_G_user = generator(test_attrs)
            p10, p20, m10, m20, g10, g20 = support.test(test_items,test_G_user.detach().numpy())
            print("epoch{}  precision_10:{:.4f},precision_20:{:.4f},map_10:{:.4f},map_20:{:.4f},ndcg_10:{:.4f},ndcg_20:{:.4f}".format(epoch, p10, p20,
                                                                                                      m10, m20, g10,
                                                                                                      g20))
            print('----------------------')
            p10_save.append(p10)
            p20_save.append(p20)
            m10_save.append(m10)
            m20_save.append(m20)
            g10_save.append(g10)
            g20_save.append(g20)

    pd.DataFrame(G_losses).to_csv('losses/G_losses.csv')
    pd.DataFrame(D_losses).to_csv('losses/D_losses.csv')
    columes = [p10_save,p20_save,m10_save,m20_save,g10_save,g20_save]
    pd.DataFrame(columes=columes).to_csv('PMG.csv')


def run():
    train()


run()






