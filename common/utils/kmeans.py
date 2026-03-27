import torch

def kmeans(X, num_clusters, max_iter=100, tol=1e-5, device='cpu', seed=100):
    """
    Args:
        X: Tensor of shape (n_samples, n_features)
        num_clusters: int
        max_iter: int
        tol: float, tolerance for convergence

    Returns:
        labels: Tensor of shape (n_samples,)
        centroids: Tensor of shape (num_clusters, n_features)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X = X.to(device)

    n_samples, n_features = X.shape

    # Randomly select the initial centroid
    random_indices = torch.randperm(n_samples, device=device)[:num_clusters]
    centroids = X[random_indices]

    for i in range(max_iter):
        # Calculate the distance from each sample to all centroids
        distances = torch.cdist(X, centroids)

        # Allocate samples to the nearest centroid
        labels = torch.argmin(distances, dim=1)

        # Update the centroid
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_clusters, device=device)

        for j in range(num_clusters):
            mask = (labels == j)
            if mask.any():
                new_centroids[j] = X[mask].mean(dim=0)
                counts[j] = mask.sum()
            else:
                # If a certain class has no samples, keep the original centroid
                pass

        # Check if it converges
        if torch.norm(centroids - new_centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids