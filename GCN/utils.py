import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(lables):
    classes = set(lables)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    lables_onehot = np.array(list(map(classes_dict.get, lables)), dtype=np.int32)
    return lables_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_lables = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_lables[:, 1:-1], dtype=np.float32)
    lables = encode_onehot(idx_features_lables[:, -1])

    # bulid graph

    idx = np.array(idx_features_lables[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(lables.shape[0], lables.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix 对称邻接矩阵
    adj = adj + adj.T.multipy(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    lables = torch.LongTensor(np.where(lables)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)


def normalize(mx):
    """Row-normalize sparse martix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, lables):
    preds = output.max(1)[1].type_as(lables)
    correct = preds.eq(lables).double()
    correct = correct.sum()
    return correct / len(lables)


def sparse_mx_to_torch_sparse_tensor(spare_mx):
    """Convert a scipt sparse matrix to a torch sparse tensor."""
    spare_mx = spare_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((spare_mx.row, spare_mx.col)).astype(np.int64))
    values = torch.from_numpy(spare_mx.data)
    shape = torch.Size(spare_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# Finished on 2022.9.9
