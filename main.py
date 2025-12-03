# financial_worldmodel.py
import os
import math
import pandas as pd
import shutil
import time
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import config

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.BATCH_SIZE
GRAD_ACCUM_STEPS = config.GRAD_ACCUM_STEPS  # Accumulate 2 steps to simulate batch_size=32
EPOCHS = config.EPOCHS
LR = config.LR
HF_AVAILABLE = True  # Enabled - 40GB GPU can handle BERT

# Feature sizes
LATENT_DIM       = config.LATENT_DIM    # Back to 256 - 40GB can handle full model
FUSED_DIM        = config.FUSED_DIM    # Back to 256 - 40GB can handle full model
TICK_FEAT_DIM    = config.TICK_FEAT_DIM     # tick feature dimension (from build_dataset)
MAX_OBS_TICKS    = config.MAX_OBS_TICKS     # Keep at 4096 to match preprocessed data
MAX_TARGET_TICKS = config.MAX_TARGET_TICKS  # Keep at 4096 to match preprocessed data

DIFFUSION_STEPS = config.DIFFUSION_STEPS
SAMPLE_STEPS = config.SAMPLE_STEPS

CKPT_DIR = "./wm_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------
# Simple DDPM scheduler (for latent diffusion)
# ---------------------------
def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class LatentDiffusionScheduler:
    def __init__(self, timesteps=DIFFUSION_STEPS, device=DEVICE):
        betas = get_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alpha_cum = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alpha_cum = alpha_cum
        self.device = device

    def q_sample(self, x_start, t, noise=None):
        # x_start: (B, latent_dim)
        if noise is None:
            noise = torch.randn_like(x_start)
        a_t = self.alpha_cum[t].view(-1, 1)
        return torch.sqrt(a_t) * x_start + torch.sqrt(1 - a_t) * noise

scheduler = LatentDiffusionScheduler()

# ---------------------------
# Dataset 
# ---------------------------
class WorldModelDataset(Dataset):
    def __init__(self, root="processed_dataset"):
        super().__init__()
        self.root = root
        self.files = sorted(os.listdir(root))  # 000000.pt 순서 정렬됨

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.files[idx])
        data = torch.load(file_path, weights_only=False)
        
        # Pre-normalize on CPU during loading (better than doing it every batch)
        # Expect obs_tick shape: (N, T, F)  (N = number of firms/assets)
        obs_tick = data["obs_tick"]         # (N, T, F)
        next_tick = data["next_tick"]       # (N, T_target, F)
        
        # Use obs stats to normalize both; compute mean/std over (N, T) so we normalize features consistently
        # result shapes will be (1,1,F) so broadcasting works for (N, T, F)
        mean = obs_tick.mean(dim=(0,1), keepdim=True)   # (1,1,F)
        std = obs_tick.std(dim=(0,1), keepdim=True) + 1e-6
        
        obs_tick_norm = ((obs_tick - mean) / std).clamp(-1e6, 1e6)
        next_tick_norm = ((next_tick - mean) / std).clamp(-1e6, 1e6)

        # Keep masks as provided. Expect obs_mask shape (N, T) boolean and obs_firm_mask shape (N,) boolean
        return {
            "obs_tick": obs_tick_norm,                 # (N, T, F)
            "obs_mask": data["obs_mask"],              # (N, T) bool-like
            "obs_firm_mask": data["obs_firm_mask"],    # (N,) bool-like (which firms are present)
            "news": data["news"],
            "next_tick": next_tick_norm,               # (N, T_target, F)
            "next_mask": data["next_mask"],            # (N, T_target)
            "next_firm_mask": data["next_firm_mask"],  # (N,)
        }



def collate_fn(batch):
    # Keep as-is for flexibility (list of dicts)
    return batch

# ---------------------------
# Encoders
# ---------------------------
# 1) TickEncoder 수정: 내부 정규화 제거 (forward에서 x = (x - x.mean...) 라인 주석/삭제)
class TickEncoder(nn.Module):
    """
    Updated to handle inputs with firm dimension:
      input x: (B, N, L, C)  where
        B = batch size
        N = number of firms/assets
        L = time ticks
        C = features per tick

    Implementation detail:
      - permute to (B, C, N, L) so Conv2d treats (N, L) as spatial dims and C as channels
      - apply Conv2d stack -> (B, hidden, N, L)
      - optional masking: mask_time (B, N, L), mask_firm (B, N)
      - global pooling over (N, L) -> (B, hidden) -> LayerNorm -> linear out
    """
    def __init__(self, in_dim=TICK_FEAT_DIM, hidden=LATENT_DIM, n_layers=4):
        super().__init__()
        layers = []
        ch = in_dim  # channels = feature dim
        for i in range(n_layers):
            # Conv2d: in_channels=ch, out_channels=hidden, kernel on (N,L)
            layers.append(nn.Conv2d(ch, hidden, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            ch = hidden
        self.net = nn.Sequential(*layers)
        # pool spatial dims (N, L) -> (1,1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, mask_time=None, mask_firm=None):
        # x: (B, N, L, C)
        # mask_time: (B, N, L)   -> per-firm per-time validity
        # mask_firm: (B, N)      -> per-firm validity (True for valid firms)
        # NOTE: remove internal normalization — do it in training loop so encoder/decoder targets agree
        # x = (x - x.mean(dim=(1,2), keepdim=True)) / (x.std(dim=(1,2), keepdim=True) + 1e-6)

        # bring features to channel dim for Conv2d
        # to (B, C, N, L)
        x = x.permute(0, 3, 1, 2).contiguous()
        h = self.net(x)  # (B, hidden, N, L)

        # apply masks if provided
        if mask_time is not None:
            mask_time = mask_time.float()
            if mask_time.dim() == 2:       # (N,L) -> (1,N,L)
                mask_time = mask_time.unsqueeze(0)
            elif mask_time.dim() == 1:     # (L,) -> (1,1,L)
                mask_time = mask_time.unsqueeze(0).unsqueeze(1)
            # h.shape = [B, C, N, L], mask_time shape = [B?, N, L] or broadcastable
            mask_time = mask_time.unsqueeze(1)  # (B,1,N,L)
            h = h * mask_time

        if mask_firm is not None:
            mask_firm = mask_firm.float()

            # tick 차원이 없으면 추가
            if mask_firm.dim() == 2:           # (B,N) -> (B,N,L)
                mask_firm = mask_firm.unsqueeze(-1).expand(-1, -1, h.shape[3])
            elif mask_firm.dim() == 3:         # (B,N,L) -> ok
                if mask_firm.shape[2] != h.shape[3]:
                    mask_firm = mask_firm.expand(-1, -1, h.shape[3])
            else:
                raise ValueError(f"Unexpected mask_firm dim: {mask_firm.shape}")

            # channel 차원 추가
            mf = mask_firm.unsqueeze(1)        # (B,1,N,L)
            h = h * mf

        p = self.pool(h).squeeze(-1).squeeze(-1)  # (B, hidden)
        p = self.norm(p)
        return self.out(p)            # (B, hidden)


class NewsEncoder(nn.Module):
    """Simple news encoder: optional HF BERT or averaged bag-of-words style."""
    def __init__(self, out_dim=LATENT_DIM, hf_model_name="distilbert-base-uncased"):
        super().__init__()
        self.out_dim = out_dim
        self.cache = {}  # Cache news embeddings to avoid recomputation
        if HF_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self.bert = AutoModel.from_pretrained(hf_model_name)
            bert_dim = self.bert.config.hidden_size
            self.proj = nn.Linear(bert_dim, out_dim)
            self.attn = nn.Linear(out_dim, 1)
        else:
            # fallback simple embedding
            self.word_emb = nn.Embedding(20000, 64)
            self.proj = nn.Linear(64, out_dim)
            self.attn = nn.Linear(out_dim,1)

    def forward(self, news_batch: List[List[str]]):
        # news_batch: list length B of list-of-strings
        B = len(news_batch)
        device = next(self.parameters()).device
        # Ensure BERT is on correct device
        if HF_AVAILABLE and self.bert.device != device:
            self.bert = self.bert.to(device)
        out_list = []
        for texts in news_batch:
            if len(texts) == 0:
                out_list.append(torch.zeros(self.out_dim, device=device))
                continue
            pieces = []
            if HF_AVAILABLE:
                for t in texts:
                    # Check cache first
                    if t in self.cache:
                        pieces.append(self.cache[t])
                    else:
                        toks = self.tokenizer(t, truncation=True, max_length=64, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = self.bert(**toks, return_dict=True)
                        cls = out.last_hidden_state[:,0,:].squeeze(0)  # (bert_dim,)
                        self.cache[t] = cls  # Cache for reuse
                        pieces.append(cls)
                stacked = torch.stack(pieces, dim=0)  # (k, bert_dim)
                reduced = self.proj(stacked)          # (k, out_dim)
            else:
                # naive: split tokens, average embeddings
                for t in texts:
                    toks = torch.randint(0, 19999, (8,), device=device)  # dummy tokens
                    emb = self.word_emb(toks).mean(dim=0)
                    pieces.append(emb)
                stacked = torch.stack(pieces, dim=0)  # (k, emb_dim)
                reduced = self.proj(stacked)
            alpha = F.softmax(self.attn(reduced).squeeze(-1), dim=0)
            pooled = (alpha.unsqueeze(-1) * reduced).sum(dim=0)
            out_list.append(pooled)
        return torch.stack(out_list, dim=0)  # (B, out_dim)

# ---------------------------
# Fusion & Latent state projection
# ---------------------------
# macro 제외 버전
class FusionNet(nn.Module):
    def __init__(self, tick_dim=LATENT_DIM, news_dim=LATENT_DIM, out_dim=LATENT_DIM):
        super().__init__()
        self.tick_proj = nn.Linear(tick_dim, out_dim)
        self.news_proj = nn.Linear(news_dim, out_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.out = nn.Linear(out_dim, out_dim)


    def forward(self, h_tick, h_news):
        # each: (B, d)
        seq = torch.stack([self.tick_proj(h_tick), self.news_proj(h_news)], dim=1)  # (B,2,d)
        fused = self.transformer(seq)  # (B,2,d)
        pooled = fused.mean(dim=1)     # (B,d)
        return self.out(pooled)        # (B,d)  -> s_t


# ---------------------------
# Latent diffusion transition model (predict noise in latent space)
# ---------------------------
class LatentDenoiser(nn.Module):
    """
    Denoiser for latent vectors: takes noisy latent x_t and condition (s_t) and predicts noise.
    Simple MLP or small Transformer.
    """
    def __init__(self, latent_dim=LATENT_DIM, cond_dim=LATENT_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, x_noisy, cond, t_emb=None):
        # x_noisy: (B, latent_dim), cond: (B, cond_dim)
        h = torch.cat([x_noisy, cond], dim=-1)
        if t_emb is not None:
            h = h + t_emb
        return self.net(h)  # predict noise

# ---------------------------
# Decoder: latent -> tick sequence
# ---------------------------
class LatentToTicksDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, out_feat=TICK_FEAT_DIM, model_dim=256, max_len=MAX_TARGET_TICKS):
        super().__init__()
        self.max_len = max_len
        self.latent_proj = nn.Linear(latent_dim, model_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=3)

        self.pos = self._build_pe(model_dim, max_len)
        self.out = nn.Linear(model_dim, out_feat)

    def _build_pe(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        i = torch.arange(0, d_model, 2).float()
        div = torch.exp(i * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def _causal_mask(self, size, device):
        # True means masked (upper triangular, excluding diagonal)
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, latent, target_len, mask=None):
        """
        latent: (B, latent_dim) -- conditioning latent (e.g., s_next or sampled z)
        target_len: int
        mask: (B, L) boolean tensor for output positions (optional)
        """
        B = latent.size(0)
        device = latent.device

        # 1) latent -> model_dim projection
        latent_proj = self.latent_proj(latent)           # (B, D)

        # 2) build memory for cross-attention: shape (B, S=1, D)
        #    The decoder will attend to this memory for conditioning.
        memory = latent_proj.unsqueeze(1)                # (B, 1, D)

        # 3) build decoder input (autoregressive target tokens)
        #    We use the latent_proj as a start-token-like embedding, repeated across T
        rep = latent_proj.unsqueeze(1).repeat(1, target_len, 1)  # (B, L, D)

        # 4) add positional encoding
        pe = self.pos[:target_len].to(device).unsqueeze(0)       # (1, L, D)
        tgt = rep + pe                                           # (B, L, D)

        # 5) causal mask to prevent attending to future positions
        tgt_mask = self._causal_mask(target_len, device)         # (L, L)

        # 6) TransformerDecoder: query=tgt, key/value=memory
        #    memory is compact (S=1) but sufficient as conditioning context
        h = self.transformer(tgt, memory=memory, tgt_mask=tgt_mask)  # (B, L, D)

        out = self.out(h)  # (B, L, feat)

        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out

# ---------------------------
# Full WorldModel container
# ---------------------------
class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tick_enc = TickEncoder()
        self.news_enc = NewsEncoder()
        # self.macro_enc = MacroEncoder()
        self.fusion = FusionNet()
        self.latent_denoiser = LatentDenoiser()
        self.decoder = LatentToTicksDecoder()
   
    def encode_obs(self, obs_tick, obs_mask, obs_firm_mask, news_list):
        """
        obs_tick: (B, N, L, C)
        obs_mask: (B, N, L)
        obs_firm_mask: (B, N)
        news_list: list-of-list (B,)
        """
        H_tick = self.tick_enc(obs_tick, mask_time=obs_mask, mask_firm=obs_firm_mask)
        H_news = self.news_enc(news_list)
        # H_macro = self.macro_enc(macro_vec)
        s_t = self.fusion(H_tick, H_news)  # (B, latent_dim)

        return s_t

    def denoise_latent(self, z_noisy, cond, t_emb=None):
        return self.latent_denoiser(z_noisy, cond, t_emb=t_emb)

    def decode(self, latent, target_len, mask=None):
        return self.decoder(latent, target_len, mask=mask)

# ---------------------------
# Time-step embedding for diffusion
# ---------------------------
def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb  # (B, dim)

# ---------------------------
# Training loop
# ---------------------------
def train():
    
    print('[*] Configure training...')
    model = WorldModel().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss = float("inf")
    best_ckpt_path = None
    patience = 4

    date_range = pd.date_range(start='2014-01-01', end = '2017-12-31', freq = "MS")

    for date in date_range :
        no_improve = 0
        
        # -----------------------------
        # 1. 데이터 전처리 > Tensor로 저장
        # -----------------------------
        print(f'[1] {date.strftime("%Y-%m")} Data Preprocessing')
        print()
        #build_tensor_process(date, 10, MAX_OBS_TICKS, TICK_FEAT_DIM)

        # -----------------------------
        # 2. Dataset loading
        # -----------------------------
        print(f'[2] Dataset Loading')
        print()
        ds = WorldModelDataset("D:\\processed_dataset")  # .pt 샘플 로딩
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,  # Increased to 8 (all CPU cores) - CPU usage is low
            collate_fn=collate_fn,
            pin_memory=True,  # Faster CPU->GPU transfer
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4  # Each worker prefetches 4 batches ahead
        )

        # -----------------------------
        # 3. Traning Loop
        # -----------------------------
        train_start_time = time.time()
        print('[3] Training Loop begins')
        print()
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.0
            total_recon = 0.0
            total_diff = 0.0
            
            batch_times = []
            for batch_idx, batch in enumerate(loader):
                batch_start = time.time()
                if batch_idx == 0:
                    print(f">>> First batch loaded, starting GPU computation...")
                    if torch.cuda.is_available():
                        print(f">>> GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                        print(f">>> GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                B = len(batch)
                # Each b["obs_tick"] is (N, L, F) -> stack -> (B, N, L, F)
                obs_tick = torch.stack([b["obs_tick"] for b in batch], dim=0).to(DEVICE, non_blocking=True)   # (B, N, L, C)
                obs_mask = torch.stack([b["obs_mask"] for b in batch], dim=0).to(DEVICE, non_blocking=True)     # (B, N, L)
                obs_firm_mask = torch.stack([b["obs_firm_mask"] for b in batch], dim=0).to(DEVICE, non_blocking=True)  # (B, N)

                next_tick = torch.stack([b["next_tick"] for b in batch], dim=0).to(DEVICE, non_blocking=True)   # (B, N, L2, C)
                next_mask = torch.stack([b["next_mask"] for b in batch], dim=0).to(DEVICE, non_blocking=True)     # (B, N, L2)
                next_firm_mask = torch.stack([b["next_firm_mask"] for b in batch], dim=0).to(DEVICE, non_blocking=True)  # (B, N)

                news_list = [b["news"] for b in batch]

                # QUICK SANITY: check NaNs/Infs in raw data
                def has_bad(t):
                    return torch.isnan(t).any().item() or torch.isinf(t).any().item()

                if has_bad(obs_tick) or has_bad(next_tick):
                    print(">>> BAD INPUT DETECTED (NaN or Inf). Dumping stats and skipping batch.")
                    print("obs_tick has_nan:", torch.isnan(obs_tick).any().item(), "has_inf:", torch.isinf(obs_tick).any().item())
                    print("next_tick has_nan:", torch.isnan(next_tick).any().item(), "has_inf:", torch.isinf(next_tick).any().item())
                    continue

                # Data is already normalized in __getitem__
                obs_tick_norm = obs_tick
                next_tick_norm = next_tick

                # encode
                if batch_idx == 0:
                    print(f">>> Encoding observations (including BERT for news)...")
                s_t = model.encode_obs(obs_tick_norm, obs_mask, obs_firm_mask, news_list)
                if batch_idx == 0:
                    print(f">>> Encoding complete, computing loss...")
                # s_next_target: encode next ticks (use both next_mask and next_firm_mask)
                s_next_target = model.tick_enc(next_tick_norm, mask_time=next_mask, mask_firm=next_firm_mask)

                # Latent diffusion loss
                t = torch.randint(0, scheduler.timesteps, (B,), device=DEVICE).long()
                noise = torch.randn_like(s_next_target)
                z_t = scheduler.q_sample(s_next_target, t, noise=noise)
                t_emb = get_timestep_embedding(t, LATENT_DIM*2).to(DEVICE)
                noise_pred = model.denoise_latent(z_t, s_t, t_emb=t_emb)
                loss_diff = F.mse_loss(noise_pred, noise, reduction='mean')

                # Reconstruction: decode s_next_target to reconstruct normalized next_tick
                # NOTE: decoder expects mask shape (B, L). But here next_mask is (B, N, L_target)
                # We reconstruct per-firm sequences aggregated? Current decoder returns (B, L, feat).
                # We will reconstruct for the *pooled* representation, so use combined mask: valid if any firm has that time valid.
                # Build time-level mask by OR over firms
                combined_next_mask = (next_mask.any(dim=1)).to(dtype=torch.bool)  # (B, L_target)

                pred_next_tick = model.decode(s_next_target, target_len=next_tick.size(2), mask=combined_next_mask)

                # For recon loss, we need to compare predicted (B, L, feat) to some target representation.
                # Original code expected next_tick to be (B, L, feat). But we currently have per-firm next_tick (B, N, L, feat).
                # A simple consistent choice: compute target as **firm-aggregated** (e.g., mean over valid firms)
                # Compute firm-weighted mean target per time:
                # next_firm_mask: (B, N) ; next_mask: (B, N, L)
                # mask for firm-time: (B, N, L)
                mask_ft = next_mask.unsqueeze(-1).float()  # (B, N, L, 1)
                # weighted sum across firms -> (B, L, feat)
                weighted = (next_tick_norm * mask_ft).sum(dim=1)  # sum over N -> (B, L, feat)
                counts = mask_ft.sum(dim=1)  # (B, L, 1)
                # avoid div by zero
                target_seq = weighted / (counts + 1e-8)  # (B, L, feat)
                # target_seq will have NaN where counts==0 -> replace with zeros
                target_seq = torch.nan_to_num(target_seq, nan=0.0, posinf=0.0, neginf=0.0)

                mask_f = combined_next_mask.unsqueeze(-1).float()  # (B, L, 1)
                # compute mean MSE only over valid positions
                sq_err = ((pred_next_tick - target_seq) ** 2) * mask_f
                recon_loss = sq_err.sum() / (mask_f.sum() * next_tick.size(-1) + 1e-8)  # mean per feature-position

                # Safety: replace NaN/Infs in intermediate tensors
                if torch.isnan(loss_diff) or torch.isnan(recon_loss) or torch.isinf(loss_diff) or torch.isinf(recon_loss):
                    print(">>> NaN/Inf in loss components. Dumping debug info and skipping update.")
                    print("loss_diff:", loss_diff, "recon_loss:", recon_loss)
                    # When debugging, print ranges (guarded)
                    try:
                        print("pred_next_tick min/max:", pred_next_tick.min().item(), pred_next_tick.max().item())
                        print("s_next_target min/max:", s_next_target.min().item(), s_next_target.max().item())
                    except Exception:
                        pass
                    continue

                # Balance losses (tune alpha/beta if needed). Start with equal weight.
                alpha = 1.0  # weight for diffusion loss
                beta = 1.0   # weight for recon loss (you may reduce this if recon dominates)
                loss = alpha * loss_diff + beta * recon_loss

                # Logging diagnostics for first few batches / or when loss huge
                if batch_idx % 10 == 0:
                    gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
                    batch_time = time.time() - batch_start
                    samples_per_sec = BATCH_SIZE / batch_time
                    print(f"[batch {batch_idx}] loss={loss.item():.6e} | GPU: {gpu_mem:.2f}GB | {batch_time:.2f}s/batch | {samples_per_sec:.1f} samples/s")
                    print(f"  loss_diff={loss_diff.item():.6e} recon={recon_loss.item():.6e}")
                    print()

                # Backprop with gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS  # Scale loss
                loss.backward()

                # Only update weights every GRAD_ACCUM_STEPS
                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    # inspect gradient norms (detect explosions)
                    total_grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_grad_norm += param_norm ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    if total_grad_norm > 1e6:
                        print(">>> Gradient explosion detected, grad_norm=", total_grad_norm)
                        # optional: skip update or scale gradients down
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.data = p.grad.data.clamp_(-1e3, 1e3)

                    # clip gradients to stabilize
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optim.step()
                    optim.zero_grad(set_to_none=True)
                
                # Save loss values before deleting tensors
                total_loss += (loss.item() * GRAD_ACCUM_STEPS)  # Unscale loss for logging
                total_recon += recon_loss.item()
                total_diff += loss_diff.item()
                
                # Delete intermediate tensors to free memory
                del loss, loss_diff, recon_loss, pred_next_tick, s_t, s_next_target, noise, z_t, noise_pred
                del target_seq, mask_ft, weighted, counts, combined_next_mask

                batch_times.append(time.time() - batch_start)
                if batch_idx == 0:
                    print(f">>> First batch took {batch_times[0]:.2f}s")
                
                # Clear cache more frequently to avoid fragmentation and memory creep
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()

            avg = total_loss / len(loader)
            epoch_end_time = time.time()
            
            # GPU memory stats
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                print(f">>> Peak GPU memory: {gpu_mem:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
            
            print("==============================")
            print(f"[{date.strftime('%Y-%m')}][Epoch {epoch+1}/{EPOCHS}] | Time = {(epoch_end_time - epoch_start_time)/60:.2f} min")
            print(f" - Avg batch time: {sum(batch_times)/len(batch_times):.3f}s")
            print(f" - Total Loss : {avg:.6f} | Recon Loss: {total_recon / len(loader):.6f} | Diff Loss: {total_diff / len(loader):.6f}")
            print("==============================")
            
            # save log
            pd.DataFrame({
                'month': date.strftime('%Y-%m'),
                'epoch': epoch+1,
                'loss': avg,
                'recon_loss': total_recon / len(loader),
                'diff_loss': total_diff / len(loader),
            }, index = [0]).to_csv('loss_log.csv', mode='a', header=not os.path.exists('loss_log.csv'), index=False)

            if avg < best_loss:
                no_improve = 0
                best_loss = avg

                ckpt_name = f"wm_{date.strftime('%Y-%m')}_best.pt"
                ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)

                torch.save(
                    {
                        "month": date.strftime('%Y-%m'),
                        "epoch": epoch+1,
                        "loss": best_loss,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                    },
                    ckpt_path
                )
                best_ckpt_path = ckpt_path
                print(f">>> Saved best checkpoint: {ckpt_path}, loss={best_loss:.6f}")

            else:
                no_improve += 1
                print(f"No improvement count: {no_improve}")

                if no_improve >= patience:
                    print(">>> Early stopping triggered.")
                    break


        train_end_time = time.time()
        print(f" - Training for {date.strftime('%Y-%m')} completed in {(train_end_time - train_start_time)/60:.2f} min")
        # -----------------------------                
        # 4. Clean up temp files
        # -----------------------------                
        shutil.rmtree("processed_dataset")
        shutil.rmtree("timespan_tick")
        shutil.rmtree("timespan_news")
        shutil.rmtree("tick")
        
        torch.cuda.empty_cache()

# ---------------------------
# Sampling / rollout
# ---------------------------
@torch.no_grad()
def sample_one_step(model: WorldModel, obs_tick, obs_mask, obs_firm_mask, news_list, target_len, steps=SAMPLE_STEPS):
    """
    Given current observation, sample s_{t+1} from latent diffusion conditioned on s_t,
    then decode to ticks.

    Inputs:
      obs_tick: (1, N, L, feat)
      obs_mask: (1, N, L)
      obs_firm_mask: (1, N)
      news_list: list-of-strings (batch size 1)
    """
    device = next(model.parameters()).device
    s_t = model.encode_obs(obs_tick.to(device), obs_mask.to(device), obs_firm_mask.to(device), [news_list])  # (1, latent)

    # Reverse diffusion sampling (ancestral DDPM)
    z = torch.randn(1, LATENT_DIM, device=device)  # start from noise
    for t_idx in reversed(range(0, scheduler.timesteps)):
        t = torch.full((1,), t_idx, dtype=torch.long, device=device)
        t_emb = get_timestep_embedding(t, LATENT_DIM*2).to(device)
        eps_pred = model.denoise_latent(z, s_t, t_emb=t_emb)  # predict noise
        beta = scheduler.betas[t_idx].item()
        alpha = scheduler.alphas[t_idx].item()
        alpha_cum = scheduler.alpha_cum[t_idx].item()

        coef1 = 1.0 / math.sqrt(alpha)
        coef2 = (beta / math.sqrt(1.0 - alpha_cum))
        mean = coef1 * (z - coef2 * eps_pred)
        if t_idx > 0:
            z = mean + math.sqrt(beta) * torch.randn_like(z)
        else:
            z = mean

    s_next = z  # (1, latent)
    generated = model.decode(s_next, target_len=target_len, mask=None)  # (1, L, feat)
    return generated.cpu()

# ---------------------------
# Example: run train (if executed as script)
# ---------------------------
if __name__ == "__main__":
    train()

    # # Example of sampling after training (pseudo)
    # model = WorldModel().to(DEVICE)
    # ck = torch.load(os.path.join(CKPT_DIR, "wm_epoch_18.pt"))
    # model.load_state_dict(ck["model"])
    # model.eval()
    # # create dummy obs
    # obs_tick = torch.randn(1, 10, MAX_OBS_TICKS, TICK_FEAT_DIM)  # (B=1, N=10, L, F)
    # obs_mask = torch.zeros(1, 10, MAX_OBS_TICKS).bool()
    # obs_mask[0, :, :200] = 1
    # obs_firm_mask = torch.ones(1, 10).bool()
    # news = ["earnings release"]
    # gen = sample_one_step(model, obs_tick, obs_mask, obs_firm_mask, news, target_len=300)
    # print("generated shape:", gen.shape)
    
    # import pdb
    # pdb.set_trace()
