import torch
import torch.nn as nn


class GPUKmeans(nn.Module):
    def __init__( 
        self, 
        K=2,
        max_iter=30, 
        stop_cond=1e-7,
    ):
        super(GPUKmeans,self).__init__()
        self.K             = K
        self.max_iter      = max_iter
        self.stop_cond     = stop_cond
      
    def all_dist(self,A,B):
        fGram  = torch.bmm(2 * B, A.transpose(1, 2))
        fNorm1 = (B * B).sum(2, keepdim=True)
        fNorm2 = (A * A).sum(2, keepdim=True)
        edm    = fNorm2.transpose(1,2) + (fNorm1 - fGram)
        return edm

    def all_dot(self,A,B):
        d = torch.bmm(A.transpose(1,2), B)
        return d

    def sel_vectors(self, X, nc):
        N = X.shape[1]
        assert nc < N, "too many clusters"
        
        # sel = np.random.permutation( N )
        sel = torch.randperm(N)[:nc]
        sel = [int(x) for x in sel]
        C = X[:, sel].clone()

        return C, sel

    def fast_gpu_kmeans(self, X, log_W, nc, verbose=False):
        
        N = X.shape[1]
        b = X.shape[0]

        C, _ = self.sel_vectors(X, nc)

        W_diff = log_W - log_W.max(dim=1)[0][:, None]
        W = W_diff.exp()
        print ("W_diff (max): {0}, W_diff (min): {1}, W (max): {2}, W (min): {3}".format(
            W_diff.max(), W_diff.min(),
            W.max(), W.min()))

        ass = 0.
        for it in range(self.max_iter):

            # C: (B, K, X), X: (B, M, X) -> D: (B, M, K)
            D = self.all_dist(C, X)
            mind , _ = torch.min(D, dim=2) # Min M for each K: (B, M, 1)

            # Select the assignment for img i, mode m to be cluster k
            # who's centroid c is of minimum distance
            ass_new  = ((D - mind[:, :, None]).abs() < 1e-8 ).type_as(X)

            df = (ass_new-ass).abs().sum(1,keepdim=True).sum(2) / N
            # percentage of points that changed assignment
            if (df < self.stop_cond).all():
                if verbose:
                    print("  gpukm converged with stop cond = %1.2e" % torch.max(df))
                break

            # # (B, M, K) one hot vectors of selections
            # ass      = ass_new * W[:, :, None]
            # # (B, 1, K), for image i how many times is each cluster k selected
            # ass_fat  = ass.sum(1, keepdim=True) 
            # # (B, M, K), for image i, cluster k, represents a distribution 
            # # (sum to 1) over the modes. E.g. torch.sum(ass_norm[0, :, 0]) = 1.0
            # ass_norm = ass / torch.clamp(ass_fat, 1e-3) 

            # ass = ass_new 
            # ass_weighted = ass * W[:, :, None]
            # # (B, 1, K), for image i how many times is each cluster k selected
            # ass_fat  = ass_weighted.sum(1, keepdim=True) 
            # # (B, M, K), for image i, cluster k, represents a distribution 
            # # (sum to 1) over the modes. E.g. torch.sum(ass_norm[0, :, 0]) = 1.0
            # ass_norm = ass_weighted / torch.clamp(ass_fat, 1e-12) 

            # (B, M, K) one hot vectors of selections
            ass = ass_new 
            # compute everything in the log domain until we have to renormalize
            # to sum to 1
            ass_weighted = ass * log_W[:, :, None] - 100. * (1 - ass)
            ass_weighted = ass_weighted - ass_weighted.max(dim=1)[0][:, None]
            ass_weighted = ass_weighted.exp() * ass
            # (B, 1, K), for image i how many times is each cluster k selected
            ass_fat  = ass_weighted.sum(1, keepdim=True) 
            # (B, M, K), for image i, cluster k, represents a distribution 
            # (sum to 1) over the modes. E.g. torch.sum(ass_norm[0, :, 0]) = 1.0
            ass_norm = ass_weighted / torch.clamp(ass_fat, 1e-12) 
            
            if (ass_fat == 0).any():
                if verbose:
                    print("warning: empty cluster")
                for bi in range(b):
                    assi = torch.nonzero(ass_fat[bi, 0]==0, as_tuple=False)[:, 0]
                    nbad = int(assi.numel())
                    if nbad > 0:
                        replace, _ = self.sel_vectors(X[bi][None], nbad)
                        X[bi:(bi+1), assi, :] = replace

            # [B, M, K]T x [B, M, X] = [B, K, M] x [B, M, X] = [B, K, X]
            C = torch.bmm(ass_norm.transpose(1, 2), X)
            
            if verbose:
                print("  gpukm (nc=%d): it %3d; df %1.2e" % (nc,it,torch.max(df)) )

            if it==self.max_iter-1 and verbose:
                print("warning: km not converged")

        # recalc objective
        D = self.all_dist(C, X)
        mind , _ = torch.min(D, dim=2)
        obj = mind.mean(dim=1)

        # calc the representatives
        mind_rep, rep_idx = torch.min(D, dim=1)
        ass_rep  = ((D - mind_rep[:, None, :]).abs() < 1e-8 ).type_as(X)
        ass_rep = ass_rep / ass_rep.sum(1, keepdim=True).clamp(1e-3)
        rep = torch.bmm(ass_rep.transpose(1, 2), X)

        return {
            'assignments': ass,
            'centroids': C,
            'representatives': rep,
            'representatives_idx': rep_idx,
            'objective': obj,
        }

    def forward(self, E, log_W=None, verbose=False, **kwargs):
        """Compute instance ids by clustering
        Args:
            E: B x N x D ... data to cluster
            W: B x N ... data to cluster
        Returns:
            results: {
                'assignments': B x N x K, # binary matrix of assignments to clusters
                'centroids': B x K x D, # cluster centroids
                'representatives': B x K x D, # nearest example to each cluster
                'representatives_idx': B x K, # index of nearest example to each cluster center
                'objective': B, # kmeans objective
            }
        """

        if log_W is None:
            W = torch.zeros_like(E[:, :, 0])
        
        results = self.fast_gpu_kmeans(
            E, log_W, self.K, verbose=verbose
        )

        return results


if __name__ == "__main__":

    dim = 52
    n_modes = 500
    ba = 10
    F = torch.randn(ba, n_modes, dim).float().cuda()

    for K in (1, 2, 10, 100):
        km = GPUKmeans(K=K)
        res = km(F, verbose=True)
        print (res)