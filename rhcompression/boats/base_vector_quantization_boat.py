import torch
import torch.nn.functional as F
from rhcore.boats.base_boat import BaseBoat
from rhtrain.utils.ddp_utils import move_to_device

def _maybe_all_reduce(t):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t)
    return t

class BaseVectorQuantizationBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        # ---- Freeze the latent encoder (AutoencoderKL) ----
        if self.models.get("latent_encoder") is not None and self.boat_config.get("freeze_latent_encoder", True):
            for p in self.models["latent_encoder"].parameters():
                p.requires_grad = False
            self.models["latent_encoder"].eval()

        # ---- VQ config / weights ----
        self.vq_weight = float(self.boat_config.get("vq_weight", 1.0))  # used only for logging totals
        self.beta = float(self.boat_config.get("commitment_beta", 0.25))  # only measured, not optimized
        self.decay = float(self.boat_config.get("ema_decay", 0.99))
        self.eps = float(self.boat_config.get("ema_eps", 1e-5))

        # quantize pre-scale (recommended for SD VAEs)
        self.quantize_pre_scale = bool(self.boat_config.get("quantize_pre_scale", True))

        # K-means warmup parameters
        self.km_target = int(self.boat_config.get("vq_kmeans_samples", 400_000))  # number of vectors to gather
        self.km_iters = int(self.boat_config.get("vq_kmeans_iters", 20))
        self._km_pool = None  # will hold [N, dim] on CPU
        self._vq_initialized = bool(self.boat_config.get("vq_skip_kmeans", False))

        # Ensure the vector quantizer exposes the buffers we will update
        vq = self.models["vector_quantizer"]
        assert hasattr(vq, "codebook") and hasattr(vq, "ema_cluster_size") and hasattr(vq, "ema_codebook"), \
            "vector_quantizer must be a VectorQuantizerNoBP exposing buffers: codebook, ema_cluster_size, ema_codebook."

        # We won't optimize VQ with an optimizer
        # If you have other optimizers in self.optimizers (e.g., for a predictor), they will still run.

        # Default loss key for logging
        if not hasattr(self, "target_loss_key"):
            self.target_loss_key = "total_loss"

        self.use_ema = False
        self.use_reference = self.validation_config.get('use_reference', True)

    # ----------------------------
    # Inference
    # ----------------------------
    @torch.no_grad()
    def predict(self, indices):
        z_q = self.dequantize_codes(indices)
        x_hat = self.decode_latents(z_q)
        return torch.clamp(x_hat, -1, 1)

    # ----------------------------
    # Training loop (no-BP)
    # ----------------------------
    def training_step(self, batch, batch_idx, epoch, *, scaler=None):
        # We won't zero/step any optimizers for VQ (no BP). Other optimizers (if present) can run as usual.
        active_keys = list(self.optimizers.keys())  # might be empty
        micro_batches = self._split_batch(batch, self.total_micro_steps)

        micro_losses_list = []
        for _, micro_batch in enumerate(micro_batches):
            micro_batch = move_to_device(micro_batch, self.device)
            micro_losses = self.training_calc_losses(micro_batch)
            micro_losses_list.append(micro_losses)

        # If you keep model EMA for other models, update it.
        self._update_ema()
        self.training_lr_scheduling_step(active_keys)

        return self._aggregate_loss_dicts(micro_losses_list)

    def training_calc_losses(self, batch):
        gt = batch["gt"]                                     # images in [-1, 1]
        vq = self.models['vector_quantizer']

        with torch.no_grad():
            enc = self.encode_images(gt)
            z = enc.latent_dist.mean if hasattr(enc, "latent_dist") else enc
            z_vq = z 

        with torch.no_grad():
            codes = vq.assign(z_vq)                          # [B, H, W]
            z_q = vq.embed_code(codes)
            self._vq_ema_update(z_vq, codes)

        # ----- Metrics (no gradients) -----
        with torch.no_grad():
            codebook_loss = F.mse_loss(z_q, z_vq)            # ||sg[z] - z_q||^2  (sg implied)
            commitment_loss = F.mse_loss(z_vq, z_q)          # ||z - sg[z_q]||^2  (both no grad)
            vq_loss = codebook_loss + self.beta * commitment_loss

            # perplexity / usage
            K = vq.codebook.size(0)
            hist = torch.bincount(codes.reshape(-1), minlength=K).float().to(z_vq.device)
            _maybe_all_reduce(hist)
            probs = hist / hist.sum().clamp_min(1)
            perplexity = torch.exp(-(probs * (probs + 1e-12).log()).sum())
            used_codes = (probs > 0).float().sum()

            # Optional latent-space MSE for monitoring
            net_loss = F.mse_loss(z_q, z_vq)

            total_loss = net_loss + self.vq_weight * vq_loss

        return {
            "total_loss": total_loss.detach(),
            "net": net_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "perplexity": perplexity.detach(),
            "used_codes": used_codes.detach(),
        }

    def validation_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        gt = batch["gt"]
        vq = self.models['vector_quantizer']

        with torch.no_grad():
            enc = self.encode_images(gt) 
            z = enc.latent_dist.mean if hasattr(enc, "latent_dist") else enc 
            z_vq = z 
            codes = vq.assign(z_vq) 
            z_q = vq.embed_code(codes) 
            z_q_dec = z_q
            x_hat = self.decode_latents(z_q_dec)
            
            valid_output = {"preds": x_hat, "targets": gt}
            metrics = self._calc_metrics(valid_output)
            named_imgs = {"groundtruth": gt, "generated": x_hat}

        return metrics, named_imgs

    def decode_latents(self, z):
        return self.models["latent_encoder"].decode(z)

    def encode_images(self, x):
        return self.models["latent_encoder"].encode(x)

    def quantize_latents(self, z):
        return self.models["vector_quantizer"].assign(z)

    def dequantize_codes(self, c: torch.Tensor) -> torch.Tensor:
        return self.models["vector_quantizer"].embed_code(c)

    @torch.no_grad()
    def _accumulate_kmeans(self, z_vq: torch.Tensor, max_take_per_batch: int = 131072):
        """
        Gather a pool of [N, dim] vectors (CPU) for k-means init.
        """
        if self._vq_initialized or self.km_target <= 0:
            return
        B, C, H, W = z_vq.shape
        zf = z_vq.permute(0,2,3,1).reshape(-1, C)  # [BHW, C]
        # take a random subset to cap memory
        N = zf.size(0)
        take = min(N, max_take_per_batch)
        idx = torch.randperm(N, device=zf.device)[:take]
        sample = zf[idx].detach().cpu()

        if self._km_pool is None:
            self._km_pool = sample
        else:
            self._km_pool = torch.cat([self._km_pool, sample], dim=0)

        if self._km_pool.size(0) >= self.km_target:
            self.models["vector_quantizer"].kmeans_init(self._km_pool.to(z_vq.device), iters=self.km_iters)
            self._km_pool = None
            self._vq_initialized = True

    @torch.no_grad()
    def _vq_ema_update(self, z_vq: torch.Tensor, codes: torch.Tensor):
        """
        EMA update of the codebook (done in-boat).
        z_vq: [B, C, H, W] in quantization space
        codes: [B, H, W] int64 assignments
        """
        vq = self.models["vector_quantizer"]

        # If k-means hasn't run yet, keep pooling samples until ready.
        if not self._vq_initialized and self.km_target > 0:
            self._accumulate_kmeans(z_vq)
            if not self._vq_initialized:
                return  # wait until k-means is done before EMA

        B, C, H, W = z_vq.shape
        zf = z_vq.permute(0,2,3,1).reshape(-1, C)          # [BHW, C]
        idx = codes.reshape(-1)                            # [BHW]
        K = vq.num_codes

        # counts and sums
        one_hot = F.one_hot(idx, K).to(zf.dtype)           # [BHW, K]
        cluster_size = one_hot.sum(0)                      # [K]
        embed_sum = one_hot.t() @ zf                       # [K, C]

        # distributed aggregate if needed
        cluster_size = _maybe_all_reduce(cluster_size)
        embed_sum = _maybe_all_reduce(embed_sum)

        # EMA update
        vq.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
        vq.ema_codebook.mul_(self.decay).add_(embed_sum,   alpha=1.0 - self.decay)

        # Normalize (with Laplace smoothing) and write codebook
        n = vq.ema_cluster_size.sum()
        smoothed = (vq.ema_cluster_size + self.eps) / (n + K * self.eps) * n

        new_cb = vq.ema_codebook / smoothed.unsqueeze(1).clamp_min(self.eps)
        vq.codebook.copy_(new_cb)
