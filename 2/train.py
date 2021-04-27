import pandas as pd
import torch
import numpy as np
import support
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Generator import *
from Discriminator import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(attr_num,attr_dim,batch_size,hidden_dim,user_emb_dim,learning_rate,alpha):
    generator = Generator(attr_num, attr_dim, hidden_dim, user_emb_dim)
    discriminator = Discriminator(attr_num, attr_dim, hidden_dim, user_emb_dim)
    generator.to(device)
    discriminator.to(device)

    print("device:", device)
    # print("Generator:")
    # print(generator)
    # print("Discriminator:")
    # print(discriminator)

    optimizerD = optim.Adam(discriminator.parameters(),
                            lr=learning_rate, weight_decay=alpha)
    optimizerG = optim.Adam(generator.parameters(),
                            lr=learning_rate, weight_decay=alpha)

    G_losses = []
    D_losses = []
    BCELoss = nn.BCELoss()
    precision10_save = []
    precision20_save = []
    map10_save = []
    map20_save = []
    ndcg10_save = []
    ndcg20_save = []

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
                    train_attr_batch, train_user_emb_batch, counter_attr_batch, counter_user_emb_batch = support.get_data(
                        index, index + batch_size)
                index = index + batch_size

                train_attr_batch.to(device)
                train_user_emb_batch.to(device)
                counter_attr_batch.to(device)
                counter_user_emb_batch.to(device)

                fake_user_emb = generator(train_attr_batch)
                fake_user_emb.to(device)

                D_real, D_logit_real = discriminator(
                    train_attr_batch, train_user_emb_batch)
                D_fake, D_logit_fake = discriminator(
                    train_attr_batch, fake_user_emb)
                D_counter, D_logit_counter = discriminator(
                    counter_attr_batch, counter_user_emb_batch)

                D_loss_real = BCELoss(D_real, torch.ones_like(D_real)).mean()
                D_loss_fake = BCELoss(D_fake, torch.zeros_like(D_fake)).mean()
                D_loss_counter = BCELoss(
                    D_counter, torch.zeros_like(D_counter)).mean()
                D_loss = D_loss_real + D_loss_fake + D_loss_counter
                optimizerD.zero_grad()

                D_loss.backward()
                optimizerD.step()
        D_losses.append(D_loss.item())

        for G_it in range(1):
            index = 0

            while index < 253236:
                if index + batch_size <= 253236:
                    train_attr_batch, _, _, _ = support.get_data(
                        index, index + batch_size)
                index = index + batch_size

                train_attr_batch.to(device)

                fake_user_emb = generator(train_attr_batch)
                fake_user_emb.to(device)
                D_fake, D_logit_fake = discriminator(
                    train_attr_batch, fake_user_emb)

                G_loss = BCELoss(D_fake, torch.ones_like(D_fake)).mean()

                optimizerG.zero_grad()
                G_loss.backward()
                optimizerG.step()

        G_losses.append(G_loss.item())

        end = time.time()
        print("epoch: {}/{}, D_loss:{:.4f}, G_loss:{:.4f}，time:{:.4f}".format(epoch,
                                                                              300, D_loss, G_loss, end-start))

        if epoch % 10 == 0:
            torch.save(generator.state_dict(),
                       'save_generator/epoch' + str(epoch) + '.pt')
            torch.save(discriminator.state_dict(),
                       'save_discriminator/epoch' + str(epoch) + '.pt')

        if epoch % 20 == 0:
            print('----------------------')
            print('BEGIN TEST:')
            test_items, test_attrs = support.get_testdata()
            test_attrs = torch.Tensor(test_attrs)
            test_attrs.to(device)

            gent = Generator()
            gent.load_state_dict(torch.load(
                'save_generator/epoch'+str(epoch)+'.pt'))
            gent.to(device)
            test_G_user = generator(test_attrs)
            precision10, precison20, map10, map20, ndcg10, ndcndcg20 = support.test(
                test_items, test_G_user.detach().numpy())
            print("epoch{}  precision_10:{:.4f},precision_20:{:.4f},map_10:{:.4f},map_20:{:.4f},ndcg_10:{:.4f},ndcg_20:{:.4f}".format(epoch, precision10, precison20,
                                                                                                                                      map10, map20, ndcg10,
                                                                                                                                      ndcndcg20))
            print('----------------------')
            precision10_save.append(precision10)
            precision20_save.append(precison20)
            map10_save.append(map10)
            map20_save.append(map20)
            ndcg10_save.append(ndcg10)
            ndcg20_save.append(ndcndcg20)

    pd.DataFrame(G_losses).to_csv('losses/G_losses.csv')
    pd.DataFrame(D_losses).to_csv('losses/D_losses.csv')
    columes = [precision10_save, precision20_save, map10_save, map20_save, ndcg10_save, ndcg20_save]
    pd.DataFrame(columes=columes).to_csv('PMG.csv')


def run(attr_num,attr_dim,batch_size,hidden_dim,user_emb_dim,learning_rate,alpha):
    train(attr_num,attr_dim,batch_size,hidden_dim,user_emb_dim,learning_rate,alpha)


if __name__ == '__main__':
    print('begin')
    attr_num = 18
    attr_dim = 5
    batch_size = 1024
    hidden_dim = 100
    user_emb_dim = 18
    learning_rate = 0.0001
    alpha = 0.0001
    run(attr_num,attr_dim,batch_size,hidden_dim,user_emb_dim,learning_rate,alpha)
