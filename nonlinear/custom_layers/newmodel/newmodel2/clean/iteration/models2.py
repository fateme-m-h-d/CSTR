import torch
import torch.nn as nn
import torch.nn.functional as F

"""Neural network models for the CSTR surrogate.

Outputs are in **scaled space** (MaxAbsScaler).

NN:
  Predicts z3 = [Ca, Cb, Cc].

NNOPT (KKThPINN-style lifted projection):
  1) NN predicts z3 = [Ca, Cb, Cc]
  2) lift to z5 = [Ca, Cb, Cc, f, g] by computing f,g from (Ca,Cb,T)
  3) apply linear KKT projection in the lifted space to satisfy A x + B z = b
  4) (optional) repeat: project -> recompute f,g -> project ... (unrolled)
     to push the result toward the true nonlinear manifold.
"""

# -----------------
# Constants (CSTR)
# -----------------
Cao = 1.0  # mol/L
Cbo = 2.0  # mol/L
Cco = 0.0  # mol/L
V = 10.0   # L
Q = 1.0    # L/s
tau = V / Q
Afo = 10e12
Eaf = 90000.0
Aro = 10e10
Ear = 80000.0
R = 8.314

device = "cpu"


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_inner_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_inner_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class NNOPT(nn.Module):
    """Lifted projection model with an *optional* unrolled consistency loop."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_num,
        z0_inner_dim,
        z0_dim,
        A,
        B,
        b,
        scaler,
        proj_iters: int = 0,
        w=None,
        debug_delta: bool = False,
    ):
        super().__init__()
        self.scaler = scaler
        self.proj_iters = int(proj_iters)
        self.debug_delta = bool(debug_delta)

        self.A = A
        self.B = B
        self.b = b

        # -----------------------------
        # Weighted KKT projector pieces
        #   z_proj = argmin ||z - z_in||_W^2  s.t.  A x + B z = b
        # chunk = M B^T (B M B^T)^{-1} where M = W^{-1}
        # -----------------------------
        if w is None:
            # Default: try to keep Ca close, let f,g move more.
            w = torch.tensor([1e6, 1.0, 1.0, 1e-6, 1e-6])
        w = w.to(device=device, dtype=self.B.dtype)
        if w.numel() != z0_dim:
            raise ValueError(f"w must have shape [{z0_dim}] but got {tuple(w.shape)}")
        if torch.any(w <= 0):
            raise ValueError("All weights w must be positive")

        M = torch.diag(1.0 / w)  # W^{-1}
        S = torch.mm(self.B, torch.mm(M, self.B.t()))  # [m,m]
        self.chunk = torch.mm(torch.mm(M, self.B.t()), torch.inverse(S))  # [z_dim,m]
        
        # self.chunk = torch.mm(B.t(),                         # B.t= transpose of B
        #                      torch.inverse(                  # inverse of a squared matrix
        #                          torch.mm(B, B.t())
        #                      )
        #                      )  

        self.Astar = -torch.mm(self.chunk, self.A)
        I = torch.eye(z0_dim, device=device, dtype=self.B.dtype)
        self.Bstar = I - torch.mm(self.chunk, self.B)
        self.bstar = torch.matmul(self.chunk, self.b).squeeze(-1)

        # Trainable NN trunk
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_inner_dim))

        # Fixed affine map implementing the projection
        self.fc_fixed1 = nn.Linear(z0_dim, z0_dim, bias=False)
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, z0_dim, bias=False)
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)

    def _compute_fg_scaled(self, T_scaled: torch.Tensor, Ca_scaled: torch.Tensor, Cb_scaled: torch.Tensor):
        """Compute (f,g) in *scaled space* to match your dataset columns."""
        # Unscale temperature and concentrations
        T = T_scaled * torch.as_tensor(self.scaler.scale_[0], device=T_scaled.device, dtype=T_scaled.dtype)
        Ca_u = Ca_scaled * torch.as_tensor(self.scaler.scale_[1], device=Ca_scaled.device, dtype=Ca_scaled.dtype)
        Cb_u = Cb_scaled * torch.as_tensor(self.scaler.scale_[2], device=Cb_scaled.device, dtype=Cb_scaled.dtype)

        # Constants as tensors
        Afo_t = torch.as_tensor(Afo, device=T.device, dtype=T.dtype)
        Eaf_t = torch.as_tensor(Eaf, device=T.device, dtype=T.dtype)
        Aro_t = torch.as_tensor(Aro, device=T.device, dtype=T.dtype)
        Ear_t = torch.as_tensor(Ear, device=T.device, dtype=T.dtype)
        R_t = torch.as_tensor(R, device=T.device, dtype=T.dtype)
        Cao_t = torch.as_tensor(Cao, device=T.device, dtype=T.dtype)
        Cbo_t = torch.as_tensor(Cbo, device=T.device, dtype=T.dtype)
        Cco_t = torch.as_tensor(Cco, device=T.device, dtype=T.dtype)

        kf = Afo_t * torch.exp(-Eaf_t / (R_t * T))
        kr = Aro_t * torch.exp(-Ear_t / (R_t * T))

        scale_f = torch.as_tensor(self.scaler.scale_[4], device=T.device, dtype=T.dtype)
        scale_g = torch.as_tensor(self.scaler.scale_[5], device=T.device, dtype=T.dtype)

        g_scaled = (kf / scale_g) * Ca_u * (Cb_u ** 2)
        f_scaled = (kr / scale_f) * (Cao_t - Ca_u + Cbo_t - Cb_u + Cco_t)
        return f_scaled, g_scaled

    def _project(self, basis_outputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.fc_fixed1(basis_outputs) + self.fc_fixed2(x)

    def forward(self, x):
        T_scaled = x
        T = T_scaled * self.scaler.scale_[0]  # unscale temperature

        # ---- Base NN prediction (scaled Ca,Cb,Cc) ----
        x0 = x
        for layer in self.layers[:-1]:
            x0 = F.relu(layer(x0))
        z0 = self.layers[-1](x0)

        Ca, Cb, Cc = z0[:, 0:1], z0[:, 1:2], z0[:, 2:3]

        def compute_fg_star_scaled(Ca_s, Cb_s):
            Ca_u = Ca_s * self.scaler.scale_[1]
            Cb_u = Cb_s * self.scaler.scale_[2]
            kf = Afo * torch.exp(-Eaf / (R * T))
            kr = Aro * torch.exp(-Ear / (R * T))
            g_star = (kf / self.scaler.scale_[5]) * Ca_u * (Cb_u ** 2)
            f_star = (kr / self.scaler.scale_[4]) * (Cao - Ca_u + Cbo - Cb_u + Cco)
            return f_star, g_star

        # initial f*, g* from NN outputs
        f_star, g_star = compute_fg_star_scaled(Ca, Cb)

        # initial lifted vector
        z5 = torch.cat([Ca, Cb, Cc, f_star, g_star], dim=1)

        n_iter = int(getattr(self, "proj_iters", 0))
        alpha = float(getattr(self, "fg_relax", 0.2))  # IMPORTANT: damping

        for k in range(max(n_iter, 0)):
            z5_in = z5

            # Step 1: project onto linear lifted constraint
            z5_proj = self.fc_fixed1(z5_in) + self.fc_fixed2(x)

            # Step 2: recompute nonlinear definitions using projected Ca,Cb
            Ca_p, Cb_p, Cc_p = z5_proj[:, 0:1], z5_proj[:, 1:2], z5_proj[:, 2:3]
            f_star, g_star = compute_fg_star_scaled(Ca_p, Cb_p)

            # Step 3: relaxed overwrite (prevents oscillation)
            f_new = (1 - alpha) * z5_proj[:, 3:4] + alpha * f_star
            g_new = (1 - alpha) * z5_proj[:, 4:5] + alpha * g_star

            z5 = torch.cat([Ca_p, Cb_p, Cc_p, f_new, g_new], dim=1)

            # ---- Debug (REAL) ----
            if getattr(self, "debug_delta", False):
                d_proj = (z5_proj - z5_in).abs().mean().item()
                d_fg   = (z5_proj[:, 3:5] - torch.cat([f_star, g_star], dim=1)).abs().mean().item()
                print(f"Iter {k}: mean|proj_step|={d_proj:.3e}  mean|fg_incons|={d_fg:.3e}")

        # Final projection (so reported output always satisfies lifted linear constraint)
        z5 = self.fc_fixed1(z5) + self.fc_fixed2(x)

        z3 = z5[:, :3]
        return z3, z5


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()
