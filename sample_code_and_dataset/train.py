import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_edge,to_edge_index,from_networkx,to_torch_coo_tensor
from torch_geometric.nn import GCNConv
import networkx as nx
from model import Encoder, drop_feature,MyGNNModel
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser
import dgl

from data_loader import load_data

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret



def label_classification(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }


acc_history = []
andrew = []
best_acc = 0
def test(model, x, edge_index, val_mask,val_labels,test_mask):
    global best_acc
    global andrew
    model.eval()
    encoder = model(x, edge_index)
    z = encoder[val_mask].detach().cpu().numpy()
    val_labels = val_labels.detach().cpu().numpy()
    test_val = encoder[test_mask].detach().cpu().numpy()
    X_train, X_test, y_train, y_test = train_test_split(z, val_labels,test_size=0.1)
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)
    #y_pred = clf.predict_proba(X_test)
    y_pred = clf.predict_proba(z)
    y_pred = torch.from_numpy(y_pred)
    _,y_pred= torch.max(y_pred,dim = 1)
    acc = accuracy_score(val_labels,y_pred)
    acc_history.append(acc)
    if acc > best_acc :
        print("store")
        z_test = encoder[test_mask].detach().cpu().numpy()
        test_ans = clf.predict_proba(z_test)
        test_ans = torch.from_numpy(test_ans)
        _,final_ans= torch.max(test_ans,dim = 1)
        andrew = final_ans
        best_acc = acc
    print(f"accuracy_score : {accuracy_score(val_labels,y_pred)}")




def train(model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    #print(f"x_1 shape {x_1.shape}, edge_index_1 shape {edge_index_1.shape}")
    z1 = model(x_1, edge_index_1).to(device)
    z2 = model(x_2, edge_index_2).to(device)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    torch.manual_seed(23344)
    random.seed(23344)

    learning_rate = 0.0005
    num_hidden = 256
    num_proj_hidden = 256
    num_layers = 2

    drop_edge_rate_1 = 0.4
    drop_edge_rate_2 = 0.1
    drop_feature_rate_1 = 0.0
    drop_feature_rate_2 = 0.2
    tau = 0.7
    num_epochs = args.epochs
    weight_decay = 0.00001

    features, graph, num_classes, train_labels, val_labels, test_labels,train_mask, val_mask, test_mask = load_data()

    features = features.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    test_labels = test_labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    adj_matrix = dgl.to_networkx(graph).to_undirected()#19717*19717
    z = nx.to_numpy_array(adj_matrix)
    edge_index = np.where(z==1)
    edge_index = torch.tensor(np.array(edge_index)).to(device)
    in_size = features.shape[1]
    out_size = num_classes

    encoder = Encoder(len(features[0]), num_hidden, k=num_layers).to(device)
    
    model = MyGNNModel(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        loss = train(model, features, edge_index)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        test(model, features, edge_index,val_mask,val_labels,test_mask)
    print("=== Final ===")
    test(model, features, edge_index,val_mask,val_labels,test_mask)
    indices = andrew
    print("Export predictions as csv file.")
    with open('output_GRACE_layer512_1500.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    #python train_gpu.py --es_iters 30 --epochs 100 --use_gpu
