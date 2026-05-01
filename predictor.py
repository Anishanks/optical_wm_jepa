"""Predictor for the optical WM JEPA.

Given z_t and a sequence of k actions, predicts the latent ẑ_{t+k}.

The GRU reads actions one-by-one starting from h_0 = f(z_t).
This naturally handles variable k without padding tricks.

For k=1: one GRU step (≈ action-conditioned linear transform of z_t)
For k=25: 25 GRU steps (recurrent rollout in latent space)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """Embed the 8-dim action vector into action_dim space.

    Action layout: [type, src, dst, slot, hop0, hop1, hop2, hop3]
    Values: type ∈ {0,1}, node indices ∈ {0..7}, slot ∈ {0..39}, hops/-1
    """

    def __init__(self, action_dim: int = 8, out_dim: int = 32):
        super().__init__()
        # Normalize: type→/1, node→/7, slot→/39, hops→/7 (or -1 → 0)
        self.register_buffer(
            'action_scale',
            torch.tensor([1., 7., 7., 39., 7., 7., 7., 7.], dtype=torch.float32)
        )
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: [B, k, 8]  (or [B, 8] for single step)
        Returns:
            [B, k, out_dim]  (or [B, out_dim])
        """
        single = actions.dim() == 2
        if single:
            actions = actions.unsqueeze(1)  # [B, 1, 8]

        # Clamp negatives (unused hops = -1) to 0 before normalizing
        a = actions.clone()
        a = torch.where(a < 0, torch.zeros_like(a), a)
        a = a / (self.action_scale + 1e-6)

        out = self.mlp(a)  # [B, k, out_dim]
        return out.squeeze(1) if single else out


class Predictor(nn.Module):
    """GRU-based predictor: (z_t, a_{t:t+k}) → ẑ_{t+k}.

    Architecture:
      1. Project z_t → h_0 (GRU initial hidden state)
      2. Encode each action in the sequence
      3. GRU reads action sequence conditioned on h_0
      4. Linear head → ẑ_{t+k}

    The GRU captures how the sequence of actions transforms the latent state.
    It naturally learns that k REMOVE steps followed by k ADD steps
    produce a different trajectory than the reverse.
    """

    def __init__(self, d_z: int = 128, action_dim: int = 8,
                 action_hidden: int = 32, gru_hidden: int = 128):
        super().__init__()
        self.d_z = d_z
        self.gru_hidden = gru_hidden

        # Project z_t → initial GRU hidden state
        self.z_to_h0 = nn.Sequential(
            nn.Linear(d_z, gru_hidden),
            nn.Tanh(),  # GRU hidden uses tanh internally
        )

        self.action_enc = ActionEncoder(action_dim, action_hidden)

        # GRU: reads action sequence, hidden is the latent state
        self.gru = nn.GRU(
            input_size=action_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Project final hidden → latent prediction
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, d_z),
            # No activation: prediction should live in the same space as z
        )

    def forward(self, z_t: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t:     [B, d_z]   latent at time t
            actions: [B, k, 8]  action sequence from t to t+k-1
        Returns:
            ẑ_{t+k}: [B, d_z]
        """
        B, k, _ = actions.shape

        # Initial hidden state from z_t
        h0 = self.z_to_h0(z_t).unsqueeze(0)  # [1, B, gru_hidden]

        # Encode action sequence
        a_emb = self.action_enc(actions)       # [B, k, action_hidden]

        # GRU rollout
        _, h_k = self.gru(a_emb, h0)          # h_k: [1, B, gru_hidden]
        h_k = h_k.squeeze(0)                   # [B, gru_hidden]

        # Project to latent space
        z_pred = self.head(h_k)                # [B, d_z]

        return z_pred

    def rollout(self, z_t: torch.Tensor,
                action_seq: torch.Tensor) -> torch.Tensor:
        """Multi-step rollout: applies all k actions at once.
        Same as forward() — exposed separately for clarity at eval time.
        """
        return self.forward(z_t, action_seq)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B, d_z = 8, 128
    pred = Predictor(d_z=d_z)

    z_t = torch.randn(B, d_z)

    # k=1
    a1 = torch.randn(B, 1, 8)
    z_pred_1 = pred(z_t, a1)
    print(f"k=1  output: {z_pred_1.shape}")  # [8, 128]

    # k=25
    a25 = torch.randn(B, 25, 8)
    z_pred_25 = pred(z_t, a25)
    print(f"k=25 output: {z_pred_25.shape}") # [8, 128]

    print(f"Predictor params: {count_params(pred):,}")  # ~115K
