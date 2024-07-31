import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utils
from sklearn import cluster
import pickle
import scipy.sparse as sparse
import time
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from metrics.cluster.accuracy import clustering_accuracy
import argparse
import random
from tqdm import tqdm
import os
import csv

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

############################################################################################

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch Siamese')
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--num_subspaces', type=int, default=10)
parser.add_argument('--gamma', type=float, default=200.0)
parser.add_argument('--lmbd', type=float, default=0.9)
parser.add_argument('--hid_dims', type=int, default=[1024, 1024, 1024])
parser.add_argument('--out_dims', type=int, default=1024)
parser.add_argument('--total_iters', type=int, default=100000)
parser.add_argument('--save_iters', type=int, default=200000)
parser.add_argument('--eval_iters', type=int, default=200000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_min', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--chunk_size', type=int, default=10000)
parser.add_argument('--non_zeros', type=int, default=1000)
parser.add_argument('--n_neighbors', type=int, default=3)
parser.add_argument('--spectral_dim', type=int, default=15)
parser.add_argument('--affinity', type=str, default="nearest_neighbor")
parser.add_argument('--mean_subtract', dest='mean_subtraction', action='store_true')
parser.set_defaults(mean_subtraction=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resume', default=None, type=str,help='resume training')
parser.add_argument('--feat', default=False, type=bool,help='get embeddings')
parser.add_argument('--ckp', default=None, type=str,help='path to load checkpoint')
parser.add_argument('--eval', default=False, type=bool,help='get embeddings')
parser.add_argument('--feat_type', default=None, type=str,help='')


global args, device
args = parser.parse_args()

if args.dataset == 'MNIST':
    args.__setattr__('gamma', 200.0)
    args.__setattr__('spectral_dim', 15)
    args.__setattr__('mean_subtract', False)
    args.__setattr__('lr_min', 0.0)
elif args.dataset == 'ColoredMNIST':
        args.__setattr__('gamma', 200)
        #args.__setattr__('num_subspaces', 10)
        #args.__setattr__('chunk_size', 10000)
        #args.__setattr__('total_iters', 50000)
        #args.__setattr__('eval_iters', 100000)
        args.__setattr__('lr_min', 0.0)
        args.__setattr__('spectral_dim', 15)
        args.__setattr__('mean_subtract', False)
        #args.__setattr__('affinity', 'symmetric')
else:
    raise Exception("Only MNIST are currently supported.")
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

################ MODEL #############################################3

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

class MLP(pl.LightningModule):
    
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)
        
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = torch.tanh_(h)
        return h


class AdaptiveSoftThreshold(pl.LightningModule):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))
    
    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class SENet(pl.LightningModule):

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=True):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims

        self.net_q = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.net_k = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        q_emb = self.net_q(queries)
        return q_emb
    
    def key_embedding(self, keys):
        k_emb = self.net_k(keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def forward(self, queries, keys):
        q = self.query_embedding(queries)
        k = self.key_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out
    
#--------------------------------------------------------------------------

class Predictor(pl.LightningModule):

    def __init__(self, input_dims, hid_dims, num_classes):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hid_dims)
        self.bn1 = nn.BatchNorm1d(hid_dims)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hid_dims, hid_dims)
        self.fc_class = nn.Linear(hid_dims, num_classes)
        self.softmax = nn.Softmax(dim=1)
       
    def get_embedding(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc_class(x)
        px = self.softmax(x)
        return x, px
    
class Projector(pl.LightningModule):

    def __init__(self, input_dims):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_dims, input_dims)
        self.bn1 = nn.BatchNorm1d(input_dims)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(input_dims, input_dims)

    def get_embedding(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        return x
############################################################################################

def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
##################################################################################################33

# Evaluate

def get_sparse_rep(senet, data, batch_size=10, chunk_size=100, non_zeros=1000):
    N, D = data.shape
    non_zeros = min(N, non_zeros)
    C = torch.empty([batch_size, N])
    if (N % batch_size != 0):
        raise Exception("batch_size should be a factor of dataset size.")
    if (N % chunk_size != 0):
        raise Exception("chunk_size should be a factor of dataset size.")

    val = []
    indicies = []
    with torch.no_grad():
        senet.eval()
        for i in range(data.shape[0] // batch_size):
            chunk = data[i * batch_size:(i + 1) * batch_size].cuda()
            q = senet.query_embedding(chunk)
            for j in range(data.shape[0] // chunk_size):
                chunk_samples = data[j * chunk_size: (j + 1) * chunk_size].cuda()
                k = senet.key_embedding(chunk_samples)   
                temp = senet.get_coeff(q, k)
                C[:, j * chunk_size:(j + 1) * chunk_size] = temp.cpu()

            rows = list(range(batch_size))
            cols = [j + i * batch_size for j in rows]
            C[rows, cols] = 0.0

            _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)
            
            val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
            index = index.reshape([-1]).cpu().data.numpy()
            indicies.append(index)
    
    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]
    
    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse


def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


def evaluate(senet, data, labels, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,
             batch_size=10000, chunk_size=10000, affinity='nearest_neighbor', knn_mode='symmetric'):
    C_sparse = get_sparse_rep(senet=senet, data=data, batch_size=batch_size,
                              chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")
    preds = utils.spectral_clustering(Aff, num_subspaces, spectral_dim)
    acc = clustering_accuracy(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds, average_method='geometric')
    ari = adjusted_rand_score(labels, preds)
    return acc, nmi, ari

##################################################################################################33

class Dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):      
        return self.data[index], self.labels[index]

class DataModule(pl.LightningDataModule):
    
    def __init__(self, data, labels,batch_size):
        super().__init__()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train = Dataset(self.data, self.labels)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        
        
##################################################################################################33

class Engine(pl.LightningModule):

    def __init__(self,model_cluster, model_bias, data, n_epochs, lr_min, n_step_per_iter, block_size, lmbd, lr=1e-4):
        super(Engine, self).__init__()
        self.model_cluster = model_cluster
        self.model_bias = model_bias
        self.learning_rate_cluster = 1e-4
        self.learning_rate_bias = 1e-3
        self.n_epochs = n_epochs
        self.lr_min = lr_min
        self.n_step_per_iter = n_step_per_iter
        self.data = data
        self.block_size = block_size
        self.lmbd = lmbd
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
    def grad_reverse(self, x):
        return GradReverse.apply(x)

    
    def training_step(self, batch_full, batch_idx):
        
        batch, labels = batch_full
        qk_labels = torch.cat((labels, labels))
        
        _lambda = 0.01
        
        opt_cluster, opt_bias= self.optimizers()
        
        opt_cluster.zero_grad()
        opt_bias.zero_grad()
        
        q_batch = self.model_cluster.query_embedding(batch)
        k_batch = self.model_cluster.key_embedding(batch)
        rec_batch = torch.zeros_like(batch).cuda()
        reg = torch.zeros([1]).cuda()
        for j in range(self.n_step_per_iter):
            block = self.data[j * self.block_size: (j + 1) * self.block_size].cuda()
            k_block = self.model_cluster.key_embedding(block)
            c = self.model_cluster.get_coeff(q_batch, k_block)
            rec_batch = rec_batch + c.mm(block)
            reg = reg + regularizer(c, self.lmbd)

        diag_c = self.model_cluster.thres((q_batch * k_batch).sum(dim=1, keepdim=True)) * self.model_cluster.shrink
        rec_batch = rec_batch - diag_c * batch
        reg = reg - regularizer(diag_c, self.lmbd)

        rec_loss = torch.sum(torch.pow(batch - rec_batch, 2))
        loss_cluster = (0.5 * args.gamma * rec_loss + reg)/args.batch_size
        
        # predict colors from feat_label. Their prediction should be uniform.
        qk_batch = torch.cat((k_batch, q_batch))
        _ , pseudo_pred_color = self.model_bias(qk_batch)
        loss_pseudo_pred_color = torch.mean(torch.sum(pseudo_pred_color*torch.log(pseudo_pred_color),1))
        
        loss = loss_cluster + loss_pseudo_pred_color*_lambda
        
        self.manual_backward(loss)
        #nn.utils.clip_grad_norm_(self.model_cluster.parameters(), 0.001)
        opt_cluster.step()
        
        opt_cluster.zero_grad()
        opt_bias.zero_grad()
        
        q_batch = self.model_cluster.query_embedding(batch)
        k_batch = self.model_cluster.key_embedding(batch)
        qk_batch = torch.cat((k_batch, q_batch))
        batch_color = self.grad_reverse(qk_batch)
        pred_color , _ = self.model_bias(batch_color)
       
        loss_color = F.cross_entropy(pred_color, qk_labels)
        
        self.manual_backward(loss_color)
        #nn.utils.clip_grad_norm_(self.model_cluster.parameters(), 0.001)
        opt_cluster.step()
        opt_bias.step()
        
        self.log_dict({"cluster":loss, "bias": loss_color},on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        return loss
    
    def configure_optimizers(self):
        opt_cluster = torch.optim.Adam(self.model_cluster.parameters(), lr=self.learning_rate_cluster)
        opt_bias = torch.optim.Adam(self.model_bias.parameters(), lr=self.learning_rate_bias)
        return opt_cluster, opt_bias

##################################################################################################33

def build_model(args, model):
    model.eval()
    model = model.cuda() 
    
    assert os.path.isfile(args.ckp), 'Error: no directory found!'
    checkpoint = torch.load(args.ckp, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = {key.replace("model_cluster.", ""): value for key, value in checkpoint['state_dict'].items() if key.split('.')[0] == 'model_cluster'}
 
    #checkpoint['state_dict'] = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    print(checkpoint['state_dict'].keys()) 
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

##################################################################################################33


def main():

    N = 15000
    
    if not args.eval:

        if not args.feat:

            fit_msg = "Experiments on {}, numpy_seed=0, total_iters=100000, lambda=0.9, gamma=200.0".format(args.dataset, args.seed)
            print(fit_msg)

            folder = "{}_result".format(args.dataset)
            if not os.path.exists(folder):
                os.mkdir(folder)

            same_seeds(args.seed)
            tic = time.time()

            #----------------------------------------------------------------------

            if args.dataset in ["MNIST"]:
                with open('datasets/{}/{}/{}_train_data.pkl'.format(args.dataset, args.feat_type, args.dataset), 'rb') as f:
                    train_samples = pickle.load(f)
                with open('datasets/{}/{}/{}_train_label.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                    train_labels = pickle.load(f)
                with open('datasets/{}/{}/{}_test_data.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                    test_samples = pickle.load(f)
                with open('datasets/{}/{}/{}_test_label.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                    test_labels = pickle.load(f)
                full_samples = np.concatenate([train_samples, test_samples], axis=0)
                full_labels = np.concatenate([train_labels, test_labels], axis=0)
            elif args.dataset in ["ColoredMNIST"]:
                #with open('datasets/{}/{}/{}_train_data.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                    #train_samples = pickle.load(f)
                #with open('datasets/{}/{}/{}_train_label_digit.pkl'.format(args.dataset,args.feat_type, args.dataset), 'rb') as f:
                    #train_labels_digit = pickle.load(f)
                #with open('datasets/{}/{}/{}_train_label_color.pkl'.format(args.dataset,args.feat_type, args.dataset), 'rb') as f:
                    #train_labels_color = pickle.load(f)
                #with open('datasets/{}/{}/{}_test_data.pkl'.format(args.dataset,args.feat_type, args.dataset), 'rb') as f:
                    #test_samples = pickle.load(f)
                #with open('datasets/{}/{}/{}_test_label_digit.pkl'.format(args.dataset,args.feat_type, args.dataset), 'rb') as f:
                    #test_labels_digit = pickle.load(f)
                #with open('datasets/{}/{}/{}_test_label_color.pkl'.format(args.dataset,args.feat_type, args.dataset), 'rb') as f:
                    #test_labels_color = pickle.load(f)
                samples = pickle.load(open('', 'rb'))
                labels_digit = pickle.load(open('', 'rb'))
                labels_color = pickle.load(open('', 'rb'))
            else:
                raise Exception("Only MNIST currently supported.")

            #----------------------------------------------------------------------

            if args.mean_subtract:
                print("Mean Subtraction")
                samples = samples - np.mean(samples, axis=0, keepdims=True)  # mean subtraction

            if args.dataset == "MNIST":
                labels = labels - np.min(labels) # 计算sre时需要label的范围是 0 ~ num_subspaces - 1
            elif args.dataset == "ColoredMNIST":
                labels_digit = labels_digit - np.min(labels_digit) # 计算sre时需要label的范围是 0 ~ num_subspaces - 1
                labels_color = labels_color - np.min(labels_color) # 计算sre时需要label的范围是 0 ~ num_subspaces - 1

            #----------------------------------------------------------------------

            #sampled_idx = np.random.choice(full_samples.shape[0], N, replace=False)
            #if args.dataset == "MNIST": 
                #samples, labels = full_samples[sampled_idx], full_labels[sampled_idx]
            #elif args.dataset == "ColoredMNIST":
                #samples, labels_digit, labels_color = full_samples[sampled_idx], full_labels_digit[sampled_idx], full_labels_color[sampled_idx] 

            block_size = min(N, 10000)

            #with open('{}/{}_{}_samples_{}.pkl'.format(folder, args.dataset, args.feat_type, N), 'wb') as f:
                #pickle.dump(samples, f)

            #if args.dataset == "MNIST":
                #with open('{}/{}_{}_labels_{}.pkl'.format(folder, args.dataset, args.feat_type, N), 'wb') as f:
                    #pickle.dump(labels, f)
            #elif args.dataset == "ColoredMNIST":
                #with open('{}/{}_{}_labels_digit_{}.pkl'.format(folder, args.dataset, args.feat_type, N), 'wb') as f:
                    #pickle.dump(labels_digit, f)
                #with open('{}/{}_{}_labels_color_{}.pkl'.format(folder, args.dataset, args.feat_type, N), 'wb') as f:
                    #pickle.dump(labels_color, f)


            all_samples, ambient_dim = samples.shape[0], samples.shape[1]

            data = torch.from_numpy(samples).float()
            data_labels = torch.from_numpy(labels_color)
            data = utils.p_normalize(data)

            n_iter_per_epoch = samples.shape[0] // args.batch_size
            n_step_per_iter = round(all_samples // block_size)
            n_epochs = args.total_iters // n_iter_per_epoch
            print(args.total_iters)

            print('EPOCHS:', n_epochs, '| ITERATIONS_PER_EPOCH:', n_iter_per_epoch, ' | STEPS_PER_ITERATION:', n_step_per_iter)

            #----------------------------------------------------------------------

            data_module = DataModule(data, data_labels, args.batch_size)
            senet = SENet(ambient_dim, args.hid_dims, args.out_dims, kaiming_init=True)
            num_classes = 3
            classifier = Predictor(args.out_dims, args.out_dims//2, num_classes)
            
            pl_model = Engine(senet, classifier, data, n_epochs, args.lr_min, n_step_per_iter, block_size, args.lmbd)

            csv_logger = pl_loggers.CSVLogger("logs_cm_v2", name=args.dataset + '_'+ args.feat_type)
            checkpoint_callback = ModelCheckpoint(monitor='cluster',filename='normcos-{epoch:02d}-{cluster:.4f}',save_top_k=30,mode='min',)

            if args.resume is not None:
                trainer = Trainer(gpus=1, strategy='dp', max_epochs=n_epochs, logger=csv_logger, resume_from_checkpoint=args.resume ,callbacks=[checkpoint_callback])
                trainer.fit(pl_model, data_module)
            else:
                trainer = Trainer(gpus=1, strategy='dp', max_epochs=n_epochs, logger=csv_logger, callbacks=[checkpoint_callback])
                trainer.fit(pl_model, data_module)

            #----------------------------------------------------------------------

        else:

            folder = "{}_result".format(args.dataset)  
            with open('{}/{}_{}_samples_{}.pkl'.format(folder, args.dataset, args.feat_type, N), 'rb') as f:
                samples = pickle.load(f)

            if args.dataset == "MNIST":
                with open('{}/{}_{}_labels_{}.pkl'.format(folder, args.dataset, args.feat_type, N), 'rb') as f:
                    labels = pickle.load(f)
            elif args.dataset == "ColoredMNIST":
                with open('{}/{}_{}_labels_digit_{}.pkl'.format(folder, args.dataset, args.feat_type, N), 'rb') as f:
                    labels = pickle.load(f)
                with open('{}/{}_{}_labels_color_{}.pkl'.format(folder, args.dataset, args.feat_type, N), 'rb') as f:
                    labels = pickle.load(f)


            # create model
            ambient_dim = samples.shape[1]
            model = build_model(args, ambient_dim)

            sampleset = Dataset(samples)
            sampleloader = DataLoader(sampleset, batch_size=100, shuffle=False, num_workers=16, pin_memory=True)

            model.eval()

            embeddings_all = []
            for x in tqdm(sampleloader, total=len(sampleloader)):
                x = x.to(device)
                embedding = model.key_embedding(x)
                embeddings_all.append(embedding.detach().cpu().numpy())

            embeddings_all = np.concatenate(embeddings_all)

            np.save(os.path.join(folder,args.dataset + '_senet_embeddings.npy'), embeddings_all)

             #----------------------------------------------------------------------

    else:
        
        folder = "{}_result".format(args.dataset)
        if not os.path.exists(folder):
            os.mkdir(folder)

        same_seeds(args.seed)
        tic = time.time()

        #----------------------------------------------------------------------

        if args.dataset in ["MNIST"]:
            #with open('datasets/{}/{}/{}_train_data.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #train_samples = pickle.load(f)
            #with open('datasets/{}/{}/{}_train_label.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #train_labels = pickle.load(f)
            #with open('datasets/{}/{}/{}_test_data.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #test_samples = pickle.load(f)
            #with open('datasets/{}/{}/{}_test_label.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #test_labels = pickle.load(f)
            #full_samples = np.concatenate([train_samples, test_samples], axis=0)
            #full_labels = np.concatenate([train_labels, test_labels], axis=0)
            test_samples = pickle.load(open('/srv/data1/ashishsingh/InvarSENnet-/SENET_v2/datasets/ColoredMNIST/scattering/ColoredMNIST_test_data_full.pkl', 'rb')) 
            test_labels = pickle.load(open('/srv/data1/ashishsingh/InvarSENnet-/SENET_v2/datasets/ColoredMNIST/scattering/ColoredMNIST_test_label_full.pkl', 'rb'))
            full_samples = test_samples
            full_labels = test_labels
        elif args.dataset in ["ColoredMNIST"]:
            #with open('datasets/{}/{}/{}_train_data.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #train_samples = pickle.load(f)
            #with open('datasets/{}/{}/{}_train_label_digit.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #train_labels_digit = pickle.load(f)
            #with open('datasets/{}/{}/{}_train_label_color.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #train_labels_color = pickle.load(f)
            #with open('datasets/{}/{}/{}_test_data.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #test_samples = pickle.load(f)
            #with open('datasets/{}/{}/{}_test_label_digit.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #test_labels_digit = pickle.load(f)
            #with open('datasets/{}/{}/{}_test_label_color.pkl'.format(args.dataset, args.feat_type,args.dataset), 'rb') as f:
                #test_labels_color = pickle.load(f)
            #full_samples = np.concatenate([train_samples, test_samples], axis=0)
            #full_labels_digit = np.concatenate([train_labels_digit, test_labels_digit], axis=0)
            #full_labels_color = np.concatenate([train_labels_color, test_labels_color], axis=0)
            full_samples = test_samples
            full_labels_digit = test_labels_digit
            full_labels_color = test_labels_color
        else:
            raise Exception("Only MNIST currently supported.")

        #----------------------------------------------------------------------

        if args.mean_subtract:
            print("Mean Subtraction")
            full_samples = full_samples - np.mean(full_samples, axis=0, keepdims=True)  # mean subtraction

        if args.dataset == "MNIST":
            full_labels = full_labels - np.min(full_labels) # 计算sre时需要label的范围是 0 ~ num_subspaces - 1
        elif args.dataset == "ColoredMNIST":
            full_labels_digit = full_labels_digit - np.min(full_labels_digit) # 计算sre时需要label的范围是 0 ~ num_subspaces - 1
            full_labels_color = full_labels_color - np.min(full_labels_color) # 计算sre时需要label的范围是 0 ~ num_subspaces - 1
            #full_labels_color = 1 - full_labels_color
            print(full_labels_color)
        #----------------------------------------------------------------------
        
        # create model
        ambient_dim = full_samples.shape[1]
        senet = SENet(ambient_dim, args.hid_dims, args.out_dims, kaiming_init=True)
        senet = build_model(args, senet)
        
        print("Evaluating on {}-full...".format(args.dataset))
        full_data = torch.from_numpy(full_samples).float()
        full_data = utils.p_normalize(full_data)
        
        if args.dataset == 'MNIST':
            acc, nmi, ari = evaluate(senet, data=full_data, labels=full_labels, num_subspaces=args.num_subspaces, affinity=args.affinity,spectral_dim=args.spectral_dim, non_zeros=args.non_zeros, n_neighbors=args.n_neighbors, batch_size=args.chunk_size,chunk_size=args.chunk_size, knn_mode='symmetric')
        elif args.dataset == 'ColoredMNIST':
            acc, nmi, ari = evaluate(senet, data=full_data, labels=full_labels_digit, num_subspaces=args.num_subspaces, affinity=args.affinity,spectral_dim=args.spectral_dim, non_zeros=args.non_zeros, n_neighbors=args.n_neighbors, batch_size=args.chunk_size,chunk_size=args.chunk_size, knn_mode='symmetric')
            
        print("N-{:d}: ACC-{:.6f}, NMI-{:.6f}, ARI-{:.6f}".format(N, acc, nmi, ari))
        #writer.writerow([N, acc, nmi, ari])
        #result.flush()
        
        #----------------------------------------------------------------------

        

if __name__ == '__main__':
    main()


