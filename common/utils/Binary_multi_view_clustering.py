import torch

def binary_multi_view_clustering(X, L=128, r=5, beta=0.003, gamma=0.01, lambda_=0.00001, init_view=0,
                                 MaxIter=5, anchor_dim=1000, n_cluster=6, innerMax=10, device='cpu'):
    """
    Implementation of the multi-view clustering algorithm in PyTorch.

    Parameters:
    X: list of torch.Tensor, where each element is a feature matrix for a view, shape (number of samples, feature dimension)
    L: int, length of the hashing code, default: 128
    r: int, power of alpha, default: 5
    beta: float, hyperparameter beta, default: 0.003
    gamma: float, hyperparameter gamma, default: 0.01
    lambda_: float, hyperparameter lambda, default: 0.00001
    MaxIter: int, maximum number of iterations
    anchor_dim: int, dimension of the anchor space
    device: str, computation device ('cuda' or 'cpu')

    Returns:
    cluster_indices: torch.Tensor, shape (number of samples,), predicted clustering labels
    """
    # Set random seeds
    torch.manual_seed(100)
    if device == 'cuda':
        torch.cuda.manual_seed(100)

    # Initialize parameters
    viewNum = len(X)  # Number of views
    N = X[0].shape[0]  # Number of samples

    # N must greater than or equal anchor_dim
    if N < anchor_dim:
        anchor_dim = N

    # Initialize random anchors for each view
    Anchor = [torch.randn(anchor_dim, x.shape[1], device=device, dtype=torch.double) for x in X]  # (anchor_dim, D)

    # Nonlinear Anchor Embedding
    scales = [0.1, 1.0, 10.0]  # multi-scale
    for v in range(viewNum):
        dist = torch.cdist(X[v], Anchor[v], p=2).pow(2)

        feaVec_list = []
        for s in scales:
            sigma = s * torch.mean(torch.min(dist, dim=1).values.sqrt())
            feaVec_s = torch.exp(-dist / (2 * sigma ** 2))
            feaVec_list.append(feaVec_s)

        feaVec = torch.mean(torch.stack(feaVec_list, dim=0), dim=0)
        X[v] = (feaVec - feaVec.mean(dim=0)).T

    # Precompute XXT
    X = torch.stack(X)  # (anchor_dim, N)
    XXT = torch.bmm(X, X.transpose(1, 2))

    # Initialize PCA-based B
    sel_sample = X[init_view][:, torch.randperm(N)[:anchor_dim]]
    U, _, _ = torch.pca_lowrank(sel_sample.T, q=L)
    B = torch.sign(torch.matmul(U[:, :L].T, X[init_view]))

    # Initialize cluster centers
    C = B[:, torch.randperm(N)[:n_cluster]]  # (L, n_cluster)
    HamDist = L - torch.matmul(B.T, C)  # (N, n_cluster)
    _, ind = HamDist.min(dim=1)
    G = torch.zeros(n_cluster, N, dtype=torch.double, device=device)
    G[ind, torch.arange(N)] = 1  # Which cluster each sample belong to
    CG = torch.matmul(C, G)  # (L, N)

    # Initialize attention weights
    alpha = torch.ones(viewNum, device=device) / viewNum

    # Main loop
    U = torch.zeros(viewNum, L, anchor_dim, dtype=torch.double, device=device)
    patience = 3
    min_delta = 1e-4
    wait = 0
    for iter in range(MaxIter):
        B_prev = B.clone().detach()

        # Update U_v
        alpha_r = alpha ** r
        UX = torch.zeros(L, N, device=device)
        for v in range(viewNum):
            A = (1 - gamma) * XXT[v] + beta * torch.eye(X[v].shape[0], dtype=torch.double, device=device).T
            Lc = torch.linalg.cholesky(A)
            y = torch.matmul(B, X[v].T).T
            U[v] = torch.cholesky_solve(y, Lc).T
            UX += alpha_r[v] * torch.matmul(U[v], X[v])

        # Update B
        B = torch.sign(UX + lambda_ * CG)
        B[B == 0] = -1

        delta_B = torch.norm(B - B_prev, p='fro').item()
        if delta_B < min_delta:
            wait += 1
            if wait >= patience:
                break
        else:
            wait = 0

        # Update C and G
        for i in range(innerMax):
            C = torch.sign(torch.matmul(B, G.T))
            C[C == 0] = 1

            rho = 0.001
            mu = 0.01

            for _ in range(3):
                grad = - torch.matmul(B, G.T) + rho * torch.sum(C, dim=0, keepdim=True).repeat(L, 1)
                C = torch.sign(C - (1 / mu) * grad)
                C[C == 0] = 1

            HamDist = L - torch.matmul(B.T, C)
            _, indx = torch.min(HamDist, dim=1)
            G.zero_()
            G[indx, torch.arange(N)] = 1

        CG = torch.matmul(C, G)

        # Update alpha
        BX = torch.matmul(U, X)  # Shape: (viewNum, L, N)
        h = (torch.norm(B - BX, p='fro', dim=(1, 2)) ** 2 -
             gamma * torch.norm(BX, p='fro', dim=(1, 2)) ** 2 +
             beta * torch.norm(U, p='fro', dim=(1, 2)) ** 2)

        H = torch.pow(h, 1 / (1 - r))
        alpha = H / torch.sum(H)

    # Get predicted labels
    _, cluster_indices = torch.max(G, dim=0)
    cluster_indices = cluster_indices.cpu().numpy()

    return cluster_indices
