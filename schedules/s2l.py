import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from schedule_base import Schedule
import numpy as np
import time
import glob

class S2L(Schedule):
    def __init__(self,
        model,
        tokenizer,
        args,
    ):
        super(S2L, self).__init__(
            model,
            tokenizer,
            args,
        )
        self.sources = np.array([data['source'] for data in self.train_data])
        self.n_sources = len(set(self.sources))
        self.distance = args["distance"]
        self.n_components = args["n_components"]
        
        if torch.distributed.get_rank() == 0:
            losses = []

            for ckpt in glob.glob(f'{args["ref_model_path"]}/*'):
                print(f"*** {ckpt} ** Loading losses...")
                try:
                    losses.append(torch.tensor(torch.load(f"{ckpt}/losses.pt")))
                except:
                    print(f"*** {ckpt} ** Could not load losses.")
                    continue
                
            if (args["num_loss_ckpts"] > -1) and (len(losses) > args["num_loss_ckpts"]):
                losses = np.stack(losses)
                keep_every = len(losses) // args["num_loss_ckpts"]
                print(f"*** Rank = {torch.distributed.get_rank()}, **Keeping every {keep_every} losses from {len(losses)} losses")
                losses = losses[np.arange(0, len(losses), keep_every)]
                self.losses = torch.from_numpy(losses).t()
            else:
                print(f"*** Rank = {torch.distributed.get_rank()}, **Using all {len(losses)} losses")
                self.losses = torch.stack(losses).t()
                print(self.losses.shape)
            # set nan to 0
            self.losses[torch.isnan(self.losses)] = 0
            
            self.losses = self.losses[self.train_idx]
            
            assert self.losses.shape[0] == self.n_pool
    
    def initialize_labeled_data(self):
        """initialize labeled data"""
        num = self.init_label_num
        if torch.distributed.get_rank() == 0:
            # rank the sources by the number of samples
            sources, counts = np.unique(self.sources, return_counts=True)
            sorted_idx = np.argsort(counts)
            
            # equally sample n samples from different sources
            sampled_indices = []
            for i in range(len(sorted_idx)):
                n_per_source = num // (len(sorted_idx) - i)
                indices = np.where(self.sources == sources[sorted_idx[i]])[0]
                if len(indices) > n_per_source:
                    new_indices = self.faiss_kmeans_selection(self.losses[indices], n_per_source)
                    sampled_indices.append(indices[new_indices])
                    num -= n_per_source
                    print(f"Sampled {n_per_source} samples from source {sources[sorted_idx[i]]} with max loss trajectory coverage")
                else:
                    sampled_indices.append(indices)
                    num -= len(indices)
                    print(f"Sampled {len(indices)} samples from source {sources[sorted_idx[i]]}")
                    
            sampled_indices = np.concatenate(sampled_indices)
                    
            self.labeled_idx[sampled_indices] = True

    def query(self, n, use_model_path):
        unlabeled_idx = torch.arange(self.n_pool)[~self.labeled_idx.bool()]  # # current unlabeled_idx
        
        # rank the sources by the number of samples
        sources, counts = np.unique(self.sources[unlabeled_idx], return_counts=True)
        sorted_idx = np.argsort(counts)
        
        # equally sample n samples from different sources
        sampled_indices = []
        for i in range(len(sorted_idx)):
            n_per_source = n // (len(sorted_idx) - i)
            indices = np.where(self.sources == sources[sorted_idx[i]])[0]
            if len(indices) > n_per_source:
                new_indices = self.faiss_kmeans_selection(self.losses[indices], n_per_source)
                sampled_indices.append(indices[new_indices])
                n -= n_per_source
                print(f"Sampled {n_per_source} samples from source {sources[sorted_idx[i]]} with max loss trajectory coverage")
            else:
                sampled_indices.append(indices)
                n -= len(indices)
                print(f"Sampled {len(indices)} samples from source {sources[sorted_idx[i]]}")
                
        sampled_indices = np.concatenate(sampled_indices)
        
        return unlabeled_idx[sampled_indices]
    
    def faiss_kmeans_selection(self, features, n):
        """
        K-means selection
        """
        import faiss
        start_time = time.time()
        kmeans = faiss.Kmeans(features.shape[1], self.n_components, niter=20, verbose=True)
        kmeans.train(features.numpy())
        
        # get the kmeans cluster labels
        D, I = kmeans.index.search(features.numpy(), 1)
        
        print(f"*** Rank = {torch.distributed.get_rank()}, **K-means took {time.time() - start_time} seconds")
        
        # get the cluster size
        clusters, counts = np.unique(I, return_counts=True)
        sorted_idx = np.argsort(counts)
        
        print(f"*** Rank = {torch.distributed.get_rank()}, **Sanple from clusters with size > 2")
        sorted_idx = sorted_idx[counts[sorted_idx] > 2]
        
        sampled_indices = []
        # sample from the largest clusters first
        for i in range(len(sorted_idx)):
            n_per_cluster = n // (len(sorted_idx) - i)
            indices = np.where(I == clusters[sorted_idx[i]])[0]
            if len(indices) > n_per_cluster:
                sampled_indices.append(np.random.choice(indices, n_per_cluster, replace=False))
                n -= n_per_cluster
            else:
                sampled_indices.append(indices)
                n -= len(indices)
                
        if n > 0:
            print(f"*** Rank = {torch.distributed.get_rank()}, **K-means: {n} samples left to sample from clusters with size <= 2")
            clusters_to_sample = clusters[np.where(counts <= 2)[0]]
            indices = np.where(np.isin(I, clusters_to_sample))[0]
            sampled_indices.append(np.random.choice(indices, n, replace=False))
                
        return np.concatenate(sampled_indices)