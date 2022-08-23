import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import DGI, LogReg
from utils import process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora", choices=("cora", "reddit", "ppi", "citeseer", "pubmed"))
    parser.add_argument("--method", default="recoverability", choices=("info_max", "random", "recoverability"))
    args = parser.parse_args()

    method = args.method
    dataset = args.dataset
    device = "cuda:0"

    # training params
    if dataset == "cora":
        batch_size = 1
        nb_epochs = 10000
        patience = 20
        lr = 0.001
        l2_coef = 0.0
        drop_prob = 0.0
        hid_units = 512
        nonlinearity = 'prelu' # special name to separate parameters
    elif dataset == "reddit": # todo : test (enable sage)
        pass
    elif dataset == "pubmed": # todo: test
        batch_size = 1
        nb_epochs = 10000
        patience = 20
        lr = 0.001
        l2_coef = 0.0
        drop_prob = 0.0
        hid_units = 512
        nonlinearity = 'prelu'  # special name to separate parameters
    elif dataset == "citeseer": # todo: test
        batch_size = 1
        nb_epochs = 10000
        patience = 20
        lr = 0.001
        l2_coef = 0.0
        drop_prob = 0.0
        hid_units = 512
        nonlinearity = 'prelu'  # special name to separate parameters
    elif dataset == "ppi": # todo: test
        batch_size = 1
        nb_epochs = 10000
        patience = 20
        lr = 0.001
        l2_coef = 0.0
        drop_prob = 0.0
        hid_units = 512
        nonlinearity = 'prelu'  # special name to separate parameters
    else:
        raise RuntimeError(f"Invalid ds: {dataset}")

    data_list, ft_size, nb_classes = process.load_data(dataset)

    model = DGI(ft_size, hid_units, nonlinearity, method)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    if method != "random":
        pbar = tqdm("Training embedding", total=nb_epochs, position=0, leave=True)
        for epoch in range(nb_epochs):
            model.train()
            agg_loss = 0
            total_nodes = 0
            for data in data_list:
                data = data.to(device)
                nb_nodes = data.x.size(0)
                features = data.x
                optimiser.zero_grad()

                idx = np.random.permutation(nb_nodes)
                shuf_fts = features[idx, :]

                lbl_1 = torch.ones(batch_size, nb_nodes)
                lbl_2 = torch.zeros(batch_size, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)
                lbl = lbl.to(device)

                logits = model(features, shuf_fts, data.edge_index)
                if method == "info_max":
                    loss = b_xent(logits, lbl)
                elif method == "recoverability":
                    loss = logits # Already computed

                agg_loss += loss.item() * nb_nodes
                total_nodes += nb_nodes

            epoch_loss = agg_loss / total_nodes
            pbar.set_description(f"Training embedding: loss: {epoch_loss:.5f}")
            pbar.update(1)

            if epoch_loss < best:
                best = epoch_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_dgi.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break

            loss.backward()
            optimiser.step()

        print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load('best_dgi.pkl'))

    with torch.no_grad():
        train_embs = []
        val_embs = []
        test_embs = []
        train_lbls = []
        val_lbls = []
        test_lbls = []
        for data in data_list:
            data = data.to(device)
            nb_nodes = data.x.size(0)
            features = data.x
            embeds = model.embed(features, data.edge_index)
            train_embs.append(embeds[data.train_mask])
            val_embs.append(embeds[data.val_mask])
            test_embs.append(embeds[data.test_mask])
            train_lbls.append(torch.argmax(data.y[data.train_mask], dim=1))
            val_lbls.append(torch.argmax(data.y[data.val_mask], dim=1))
            test_lbls.append(torch.argmax(data.y[data.test_mask], dim=1))

        train_embs = torch.cat(train_embs)
        val_embs = torch.cat(val_embs)
        test_embs = torch.cat(test_embs)
        train_lbls = torch.cat(train_lbls)
        val_lbls = torch.cat(val_lbls)
        test_lbls = torch.cat(test_lbls)

    tot = torch.zeros(1)
    tot = tot.to(device)

    accs = []

    pbar = tqdm("Training classifier", total=50, position=0, leave=True)
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.to(device)

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.to(device)
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        #print(acc.item())
        pbar.set_description(f"Training classifier: Accuracy:{acc.item():.5}")
        pbar.update(1)
        tot += acc

    #print('Average accuracy:', tot / 50)
    pbar.close()
    sys.stdout.flush()
    accs = torch.stack(accs)
    print(f"Average acc: {accs.mean().item()}")
    print(f"Acc STD: {accs.std().item()}")
    print(f"Acc [min, max]: [{torch.min(accs).item()}, {torch.max(accs).item()}]")

