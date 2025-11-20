import torch
import torch.nn as nn
import torch.nn.functional as F

class DVAE(nn.Module):
    """
    DVAE complet :
      - modèle génératif p_θ(z_t | h_t^p, u_t), p_θ(x_t | h_t^p, z_t)
      - modèle d'inférence q_φ(z_t | h_t^q, u_t)
      - mêmes fonctions que DVAE_GenerativeRNN1 :
          * step  (pour le prior, utilisé en génération)
          * forward(x, u)  (teacher forcing, utilise q_φ)
          * generate(T, u, x0=None)
    """

    def __init__(self, x_dim, z_dim, u_dim, h_dim=128):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.h_dim = h_dim

        # ==========================
        # 1) RNN génératif (prior) : h_t^p
        # ==========================
        self.d_h = nn.GRU(input_size=x_dim, hidden_size=h_dim, batch_first=True)

        # p_θ(z_t | h_t^p, u_t)
        self.p_z = nn.Sequential(
            nn.Linear(h_dim + u_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * z_dim)   # [mu_p, log_sigma_p]
        )

        # p_θ(x_t | h_t^p, z_t)
        self.d_x = nn.Sequential(
            nn.Linear(h_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * x_dim)   # [mu_x, log_sigma_x]
        )

        # ==========================
        # 2) RNN d'inférence (posterior) : h_t^q
        # ==========================
        self.e_h = nn.GRU(input_size=x_dim, hidden_size=h_dim, batch_first=True)

        # q_φ(z_t | h_t^q, u_t)
        self.q_z = nn.Sequential(
            nn.Linear(h_dim + u_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * z_dim)   # [mu_q, log_sigma_q]
        )

    # --------------------------------------------------------
    # KL divergence entre deux Gaussiennes diag
    # --------------------------------------------------------
    @staticmethod
    def kl_normal(mu_q, log_sigma_q, mu_p, log_sigma_p):
        """
        KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) )
        mu_*, log_sigma_* : (B,1,z_dim)
        """
        sigma_q2 = torch.exp(2 * log_sigma_q)
        sigma_p2 = torch.exp(2 * log_sigma_p)

        term1 = (sigma_q2 + (mu_q - mu_p) ** 2) / sigma_p2
        term2 = 2 * (log_sigma_p - log_sigma_q)
        kl = 0.5 * (term1 + term2 - 1)
        return kl.sum(dim=-1)  # somme sur dim latent → (B,1)

    # --------------------------------------------------------
    # Helper: un pas de PRIOR (génération) — comme ton ancien step
    # --------------------------------------------------------
    def step(self, x_prev, h_prev, u_t):
        """
        Pas génératif avec le PRIOR (utilisé en génération)
        x_prev : (B,1,x_dim)
        h_prev : (1,B,h_dim)
        u_t    : (B,1,u_dim)
        """
        # h_t^p
        out_hp, h_t = self.d_h(x_prev, h_prev)   # (B,1,h_dim)

        # p_θ(z_t | h_t^p, u_t)
        p_in = torch.cat([out_hp, u_t], dim=-1)
        p_params = self.p_z(p_in)
        mu_p, log_sigma_p = torch.chunk(p_params, 2, dim=-1)

        sigma_p = torch.exp(log_sigma_p)
        eps = torch.randn_like(mu_p)
        z_t = mu_p + sigma_p * eps

        # p_θ(x_t | h_t^p, z_t)
        x_in = torch.cat([out_hp, z_t], dim=-1)
        x_params = self.d_x(x_in)
        mu_x, log_sigma_x = torch.chunk(x_params, 2, dim=-1)

        return mu_p, log_sigma_p, z_t, mu_x, h_t

    # --------------------------------------------------------
    # 1) FORWARD : teacher forcing + inference q_φ
    #     -> retourne x_hat, z, KL_total
    # --------------------------------------------------------
    def forward(self, x, u):
        """
        x : (B,T,x_dim)
        u : (B,T,u_dim)

        Utilise q_φ(z_t | x_t, u_t) pour l'entraînement.
        Retourne :
            z_samples : (B,T,z_dim)
            x_mu      : (B,T,x_dim)
            kl_sum    : scalaire (somme KL sur batch et temps)
        """
        B, T, _ = x.shape
        device = x.device

        # États init
        h_p = torch.zeros(1, B, self.h_dim, device=device)  # prior RNN
        h_q = torch.zeros(1, B, self.h_dim, device=device)  # encoder RNN
        x_prev = torch.zeros(B, 1, self.x_dim, device=device)

        z_list = []
        x_mu_list = []
        kl_list = []

        for t in range(T):
            u_t = u[:, t:t+1, :]        # (B,1,u_dim)
            x_t = x[:, t:t+1, :]        # (B,1,x_dim)

            # ------- RNN génératif (prior) : h_t^p -------
            out_hp, h_p = self.d_h(x_prev, h_p)  # (B,1,h_dim)

            # ------- RNN encodeur (posterior) : h_t^q -----
            out_hq, h_q = self.e_h(x_t, h_q)     # (B,1,h_dim)

            # ------- Prior p_θ(z_t | h_t^p, u_t) ----------
            p_in = torch.cat([out_hp, u_t], dim=-1)
            p_params = self.p_z(p_in)
            mu_p, log_sigma_p = torch.chunk(p_params, 2, dim=-1)

            # ------- Posterior q_φ(z_t | h_t^q, u_t) ------
            q_in = torch.cat([out_hq, u_t], dim=-1)
            q_params = self.q_z(q_in)
            mu_q, log_sigma_q = torch.chunk(q_params, 2, dim=-1)

            # ------- Sampling z_t ~ q_φ -------------------
            sigma_q = torch.exp(log_sigma_q)
            eps = torch.randn_like(mu_q)
            z_t = mu_q + sigma_q * eps

            # ------- Decoder p_θ(x_t | h_t^p, z_t) --------
            x_in = torch.cat([out_hp, z_t], dim=-1)
            x_params = self.d_x(x_in)
            mu_x, log_sigma_x = torch.chunk(x_params, 2, dim=-1)

            # ------- KL_t = KL(q_φ || p_θ) ----------------
            kl_t = self.kl_normal(mu_q, log_sigma_q, mu_p, log_sigma_p)  # (B,1)

            # stockage
            z_list.append(z_t)
            x_mu_list.append(mu_x)
            kl_list.append(kl_t)

            # teacher forcing : on donne le vrai x_t au pas suivant
            x_prev = x_t

        z_samples = torch.cat(z_list, dim=1)      # (B,T,z_dim)
        x_mu = torch.cat(x_mu_list, dim=1)        # (B,T,x_dim)
        kl = torch.cat(kl_list, dim=1)            # (B,T)
        kl_sum = kl.mean()                        # moyenne sur batch et temps

        return z_samples, x_mu, kl_sum

    # --------------------------------------------------------
    # 2) GÉNÉRATION AUTONOME (free-running) — comme avant
    #     Utilise le PRIOR uniquement
    # --------------------------------------------------------
    def generate(self, T, u, x0=None):
        """
        Génère un signal de longueur T en mode autonome.
        u  : (B,T,u_dim)
        x0 : (B,x_dim) ou None
        """
        B = u.size(0)
        device = u.device

        h_p = torch.zeros(1, B, self.h_dim, device=device)

        if x0 is None:
            x_prev = torch.zeros(B, 1, self.x_dim, device=device)
        else:
            x_prev = x0.unsqueeze(1)

        z_list = []
        x_list = []

        for t in range(T):
            u_t = u[:, t:t+1, :]

            mu_p, log_sigma_p, z_t, mu_x, h_p = self.step(x_prev, h_p, u_t)

            z_list.append(z_t)
            x_list.append(mu_x)

            # génération autonome : on réinjecte la sortie générée
            x_prev = mu_x

        return torch.cat(z_list, dim=1), torch.cat(x_list, dim=1)
