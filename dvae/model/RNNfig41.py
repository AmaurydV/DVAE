import torch
import torch.nn as nn
import torch.nn.functional as F

class DVAE_Generative(nn.Module):
    """
    Implémentation EXACTE des équations (4.9)–(4.13)
    + prédiction + génération autonome
    """

    def __init__(self, x_dim, z_dim, u_dim, h_dim=128):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.h_dim = h_dim

        # ----------------------------------------------------
        # (4.9) h_t = d_h(x_{t-1}, h_{t-1})
        # ----------------------------------------------------
        self.d_h = nn.GRU(input_size=x_dim, hidden_size=h_dim, batch_first=True)

        # ----------------------------------------------------
        # (4.10) d_z(h_t, u_t) → μ_z, σ_z
        # ----------------------------------------------------
        self.d_z = nn.Sequential(
            nn.Linear(h_dim + u_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * z_dim)
        )

        # ----------------------------------------------------
        # (4.12) d_x(h_t, z_t) → μ_x, σ_x
        # ----------------------------------------------------
        self.d_x = nn.Sequential(
            nn.Linear(h_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * x_dim)
        )


    # --------------------------------------------------------
    # Helper: one generative step
    # --------------------------------------------------------
    def step(self, x_prev, h_prev, u_t):
        """
        Réalise UN PAS de génération du DVAE
        x_prev : B × 1 × x_dim
        h_prev : 1 × B × h_dim
        u_t    : B × 1 × u_dim
        """

        # (4.9) h_t = d_h(x_{t-1}, h_{t-1})
        out_h, h_t = self.d_h(x_prev, h_prev)

        # (4.10) d_z(h_t, u_t)
        z_input = torch.cat([out_h, u_t], dim=-1)
        z_params = self.d_z(z_input)
        mu_z, log_sigma_z = torch.chunk(z_params, 2, dim=-1)
        sigma_z = torch.exp(log_sigma_z)

        # (4.11) z_t ~ N(mu_z, sigma_z)
        eps = torch.randn_like(mu_z)
        z_t = mu_z + sigma_z * eps

        # (4.12) d_x(h_t, z_t)
        x_input = torch.cat([out_h, z_t], dim=-1)
        x_params = self.d_x(x_input)
        mu_x, log_sigma_x = torch.chunk(x_params, 2, dim=-1)

        return mu_z, sigma_z, z_t, mu_x, h_t
    

    # --------------------------------------------------------
    # 1) PRÉDICTION (teacher forcing)
    # --------------------------------------------------------
    def forward(self, x, u):
        """
        x : B × T × x_dim  (signal réel)
        u : B × T × u_dim
        """

        B, T, _ = x.shape

        h = torch.zeros(1, B, self.h_dim, device=x.device)
        x_prev = torch.zeros(B, 1, self.x_dim, device=x.device)

        z_list = []
        x_pred_list = []

        for t in range(T):
            u_t = u[:, t:t+1, :]

            # un pas complet
            mu_z, sigma_z, z_t, mu_x, h = self.step(x_prev, h, u_t)

            # stockage
            z_list.append(z_t)
            x_pred_list.append(mu_x)

            # teacher forcing → utiliser le vrai x_t
            x_prev = x[:, t:t+1, :]

        return torch.cat(z_list, dim=1), torch.cat(x_pred_list, dim=1)


    # --------------------------------------------------------
    # 2) GÉNÉRATION AUTONOME (free-running)
    # --------------------------------------------------------
    def generate(self, T, u, x0=None):
        """
        Génère un signal longueur T
        u : B × T × u_dim
        x0 : état initial (optionnel)
        """

        B = u.size(0)

        h = torch.zeros(1, B, self.h_dim, device=u.device)

        if x0 is None:
            x_prev = torch.zeros(B, 1, self.x_dim, device=u.device)
        else:
            x_prev = x0.unsqueeze(1)

        z_list = []
        x_list = []

        for t in range(T):
            u_t = u[:, t:t+1, :]

            # un pas complet
            mu_z, sigma_z, z_t, mu_x, h = self.step(x_prev, h, u_t)

            # stockage
            z_list.append(z_t)
            x_list.append(mu_x)

            # génération autonome = réinjecter la sortie
            x_prev = mu_x

        return torch.cat(z_list, dim=1), torch.cat(x_list, dim=1)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# ==========================================================
# 2) LOSS simple (MSE)
# ==========================================================
def reconstruction_loss(x_true, x_pred):
    return nn.MSELoss()(x_pred, x_true)

# ==========================================================
# 3) TRAIN LOOP
# ==========================================================
def train(model, dataloader, optimizer, epochs=50, device="cuda"):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for x, u in dataloader:
            x = x.to(device)
            u = u.to(device)

            # forward (teacher forcing)
            z_pred, x_pred = model(x, u)

            # reconstruction loss
            loss = reconstruction_loss(x, x_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss = {total_loss/len(dataloader):.6f}")