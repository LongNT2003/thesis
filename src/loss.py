import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: (batch_size, embedding_dim) - Embedding of original images
        z_j: (batch_size, embedding_dim) - Embedding of posittive images
        """
        batch_size = z_i.shape[0]

        # Normalize the embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings to form a (2N, embedding_dim) tensor
        z = torch.cat([z_i, z_j], dim=0)

        # Compute cosine similarity matrix (scaled by temperature)
        sim = torch.matmul(z, z.T) / self.temperature  # shape: (2N, 2N)

        # Mask out self-similarities by replacing diagonal with a very low value
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # For each sample i, the positive sample is at index (i + batch_size) mod (2N)
        positive_indices = (
            torch.arange(2 * batch_size, device=z.device) + batch_size
        ) % (2 * batch_size)
        positives = sim[torch.arange(2 * batch_size), positive_indices].unsqueeze(1)

        # Remove the positive column from sim to obtain negatives for each sample.
        all_indices = (
            torch.arange(2 * batch_size, device=z.device)
            .unsqueeze(0)
            .expand(2 * batch_size, -1)
        )
        pos_indices = positive_indices.unsqueeze(1)
        neg_mask = all_indices != pos_indices
        negatives = sim[neg_mask].view(2 * batch_size, -1)

        # Construct logits: first column is the positive, remaining are negatives.
        logits = torch.cat([positives, negatives], dim=1)

        # Labels: the positive is at index 0 for each sample.
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class NTXentLossHardNegatives(torch.nn.Module):
    def __init__(self, temperature=0.5, top_k_negatives=None):
        """
        Args:
            temperature: Scaling factor for similarities.
            top_k_negatives (int, optional): If not None, use only the top K
                                             hardest negatives per anchor.
                                             Defaults to None (use all negatives).
        """
        super(NTXentLossHardNegatives, self).__init__()
        self.temperature = temperature
        self.top_k_negatives = top_k_negatives
        # Sử dụng giá trị âm vô cùng nhỏ để đảm bảo không được chọn bởi topk
        self.mask_value = -float("inf")

    def forward(self, z_i, z_j):
        """
        z_i: (batch_size, embedding_dim) - Embedding of product images (or reviews)
        z_j: (batch_size, embedding_dim) - Embedding of corresponding review images (or products)
        """
        device = z_i.device
        batch_size = z_i.shape[0]

        # 1. Normalize the embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 2. Concatenate embeddings: [p1..pn, r1..rn]
        z = torch.cat([z_i, z_j], dim=0)  # shape: (2N, embedding_dim)
        n_samples = z.shape[0]  # Should be 2 * batch_size

        # 3. Compute cosine similarity matrix (scaled by temperature)
        sim = torch.matmul(z, z.T) / self.temperature  # shape: (2N, 2N)

        # 4. Create masks
        # Mask for self-similarities (diagonal)
        self_mask = torch.eye(n_samples, dtype=torch.bool, device=device)

        # Mask for positive pairs
        # Positive for pi (at index i) is ri (at index i + batch_size)
        # Positive for ri (at index i + batch_size) is pi (at index i)
        pos_indices = (torch.arange(n_samples, device=device) + batch_size) % n_samples
        # Create a full mask for positives for easier indexing later
        pos_mask = torch.zeros_like(self_mask, dtype=torch.bool)
        pos_mask[torch.arange(n_samples), pos_indices] = True

        # 5. Extract positive similarities
        # Use the pos_indices calculated earlier
        positives = sim[torch.arange(n_samples), pos_indices].unsqueeze(
            1
        )  # shape: (2N, 1)

        # 6. Extract negative similarities and apply hard negative mining
        # Start with the full similarity matrix
        negatives_sim = sim.clone()

        # Mask out self-similarities and positives so they aren't selected as negatives
        negatives_sim.masked_fill_(self_mask, self.mask_value)
        negatives_sim.masked_fill_(pos_mask, self.mask_value)

        if self.top_k_negatives is not None:
            # Ensure k is not larger than the number of actual negatives available
            num_actual_negatives = n_samples - 2  # Exclude self and positive
            k_to_use = min(self.top_k_negatives, num_actual_negatives)

            if k_to_use > 0:
                # Select the top k highest similarity values (hardest negatives) for each row
                # topk returns values and indices, we only need values
                hard_negatives, _ = torch.topk(
                    negatives_sim, k_to_use, dim=1, largest=True
                )
            else:
                # Handle edge case where k=0 or no negatives available (e.g., batch_size=1)
                hard_negatives = torch.empty((n_samples, 0), device=device)

            negatives = hard_negatives  # Use only the hardest ones
        else:
            # If top_k_negatives is None, use all available negatives
            # We can extract them using the combined mask
            # This branch makes it equivalent to the original NTXentLoss logic,
            # but extracting via topk might be computationally similar anyway.
            # Let's extract them directly for clarity when not using topk.
            negative_mask = ~(self_mask | pos_mask)
            # Need to gather carefully, perhaps sticking with topk is simpler?
            # Let's stick to topk logic for consistency, setting k to max possible negatives.
            num_actual_negatives = n_samples - 2
            if num_actual_negatives > 0:
                negatives, _ = torch.topk(
                    negatives_sim, num_actual_negatives, dim=1, largest=True
                )
            else:
                negatives = torch.empty((n_samples, 0), device=device)

        # 7. Construct logits
        # First column is the positive similarity, subsequent columns are negative similarities
        logits = torch.cat(
            [positives, negatives], dim=1
        )  # shape: (2N, 1 + k_to_use or 1 + 2N-2)

        # 8. Create labels
        # The positive similarity is always at index 0
        labels = torch.zeros(n_samples, dtype=torch.long, device=device)

        # 9. Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
