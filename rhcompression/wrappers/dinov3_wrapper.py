import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from rhcompression.wrappers.cache_mgmt.two_tier import TwoTierFeatureCache
import time

from einops import rearrange

class Dinov3Wrapper:
    """
    Thin wrapper to extract DINOv3 hidden features with robust input-range handling + optional two-tier LRU cache.

    Args:
        model_name: Hugging Face model id, e.g. "facebook/dinov3-vitb16-pretrain-lvd1689m"
                    or "facebook/dinov3-convnext-base-pretrain-lvd1689m".
        input_range: Tuple (min, max) describing *your* x's value range. Examples:
                     (0,1), (-1,1), (0,255). Used only when x is a tensor/ndarray.
        device: Torch device string or torch.device. If None, inferred from model.
        dtype: Optional torch dtype to cast model & inputs (e.g., torch.float16, torch.bfloat16).
        image_size: Optional override for the processor's resize (int). If None, use checkpoint default.
        trust_remote_code: Forwarded to HF loaders if you use custom builds.

        enable_cache: bool (default True)
        mem_cache_bytes: int (default 32GB)
        disk_cache_dir: path (default ~/.cache/dinov3_feature_cache)
        disk_cache_bytes: int | None (None = unlimited)
        cache_hidden_states: bool (default False; they can be very large)

    Methods:
        get_features(x, output_hidden_states=False) -> dict with:
            - last_hidden_state: torch.FloatTensor
              * ViT: [B, 1 + num_register_tokens + Npatch, D]
              * ConvNeXt: [B, C, H', W']
            - pooler_output: torch.FloatTensor [B, D] (global feature)
            - cls_token: torch.FloatTensor [B, D] (ViT only; None for ConvNeXt)
            - patch_tokens: torch.FloatTensor [B, Npatch, D] (ViT) or [B, C, H', W'] (ConvNeXt)
            - hidden_states: tuple of layer outputs (if output_hidden_states=True)
    """

    def __init__(self, model_name, input_range=(-1.0, 1.0), 
                 dtype=None, image_size=None, trust_remote_code=False,
                 enable_cache=True, mem_cache_in_gb=16,
                 disk_cache_dir=None, disk_cache_bytes=None, cache_hidden_states=False):
        
        self.model_name = model_name
        self.input_min, self.input_max = input_range
        if self.input_max <= self.input_min:
            raise ValueError(f"Invalid input_range {input_range}. Must satisfy max > min.")

        # Image processor carries the correct resize + normalization (mean/std) for the checkpoint
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, do_resize=False
        )

        mem_cache_bytes = int(mem_cache_in_gb * (1024**3))

        # Optionally override size if you want something different than the checkpoint default
        self._override_size = image_size

        # Load the backbone
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # dtype / device
        if dtype is not None:
            self.model.to(dtype=dtype)

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        # Cache
        self._cache = None
        if enable_cache:
            self._cache = TwoTierFeatureCache(
                mem_limit_bytes=mem_cache_bytes,
                disk_limit_bytes=disk_cache_bytes,
                cache_dir=disk_cache_dir,
                cache_hidden_states=cache_hidden_states,)

    @property
    def device(self):
        # robust way to get current model device (handles sharded/device_map too)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    
    def to(self, device):
        """Move model to device."""
        self.model.to(device)

        return self

    def _to_tensor_batch(self, x):
        """
        If x is a single tensor/ndarray/image, wrap into a list to make batching consistent.
        """
        if isinstance(x, (torch.Tensor, np.ndarray, Image.Image)):
            return [x]
        return x  # assume iterable of images

    def _renormalize_to_unit(self, arr):
        """
        Map arbitrary input range -> [0,1] float tensor in CHW.
        Supports:
          - torch.Tensor in [B,C,H,W] or [C,H,W] or [H,W,C]
          - numpy arrays same shapes
          - uint8 is handled as (0,255) regardless of input_range
        """
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)

        # Infer data format; accept HWC or CHW
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
            tensor = arr
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):  # HWC
            tensor = arr.permute(2, 0, 1)
        elif arr.ndim == 4 and arr.shape[1] in (1, 3):  # BCHW
            tensor = arr
        elif arr.ndim == 4 and arr.shape[-1] in (1, 3):  # BHWC
            tensor = arr.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"Unsupported tensor/array shape {tuple(arr.shape)}; expected CHW or HWC (or batched)."
            )

        tensor = tensor.to(torch.float32)

        # If already uint8 (via ndarray) you'll be in float now but values ∈ [0..255]; detect via original dtype if needed.
        # Rule: if max>1.5*255 we clamp, else if it looks like 0..255 we divide, else use provided input_range.
        vmin, vmax = float(tensor.min()), float(tensor.max())
        if vmax > 260.0:  # clearly not image-like
            # fall back to declared range
            unit = (tensor - self.input_min) / (self.input_max - self.input_min)
        elif vmax > 1.5:  # assume 0..255-style
            unit = tensor / 255.0
        else:
            # use declared input_range, e.g. (-1,1) or (0,1)
            unit = (tensor - self.input_min) / (self.input_max - self.input_min)

        return unit.clamp_(0.0, 1.0)

    def _prepare_inputs(self, images):
        """
        Convert arbitrary inputs to the checkpoint's expected pixel_values using the official image processor.
        We convert tensors/ndarrays to [0,1] ourselves and set do_rescale=False so the processor only
        applies resize + mean/std normalization from the checkpoint config. :contentReference[oaicite:2]{index=2}
        """
        images = self._to_tensor_batch(images)

        # We'll build a mixed list: PIL/np/torch are all accepted by the processor.
        prepped: list = []
        use_manual_rescale_flags: list[bool] = []

        for img in images:
            if isinstance(img, (torch.Tensor, np.ndarray)):
                prepped.append(self._renormalize_to_unit(img))
                use_manual_rescale_flags.append(True)   # already in [0,1]
            elif isinstance(img, Image.Image):
                prepped.append(img)                     # let processor rescale from [0..255]
                use_manual_rescale_flags.append(False)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

        # If *any* item is already [0,1], we call processor with do_rescale=False.
        # Otherwise, we let it rescale from [0..255] → [0,1].
        do_rescale = not any(use_manual_rescale_flags)

        size_kw = {}
        if self._override_size is not None:
            # DINOv3 ViT uses square resize; ConvNeXt accepts square as well.
            size_kw = {"size": self._override_size}

        batch = self.processor(
            images=prepped,
            return_tensors="pt",
            do_rescale=do_rescale,
            **size_kw,
        )
        # Move to model device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, dtype=next(self.model.parameters()).dtype)
        return batch

    def _make_key(self, id, output_hidden_states):
        """
        Prefer an explicit user-provided id. If None, create a lightweight fingerprint
        based on model name + image size + output flag and a monotonic counter.
        (Avoid hashing full image tensors to keep things fast.)
        """
        base = str(id) if id is not None else f"auto-{time.time_ns()}"
        model_type = getattr(self.model.config, "model_type", "")
        over = str(self._override_size) if self._override_size is not None else "default"
        return f"{self.model_name}|{model_type}|size={over}|hs={int(bool(output_hidden_states))}|{base}"

    @torch.inference_mode()
    def get_features(self, x, x_raw_keys, output_hidden_states: bool = False):
        """
        Run the backbone and return a tidy tensor of features in [B, C, H', W'].
        - ViT-like (e.g., DINOv3 ViT): returns patch tokens rearranged to [B, D, H/16, W/16]
        - ConvNeXt-like: returns the spatial map as-is from last_hidden_state
        """

        B, C, H, W = x.shape
        sH = H // 16
        sW = W // 16

        # Prepare hit/miss bookkeeping
        results = [None] * B              # each entry becomes a [C, sH, sW] tensor
        miss_ids, miss_keys = [], []

        for i, x_raw_key in enumerate(x_raw_keys):
            key = self._make_key(x_raw_key, output_hidden_states)
            hit = None
            if getattr(self, "_cache", None) is not None:
                hit = self._cache.get(key, device=self.device)

            if hit is not None:
                # print("Cache hit for key:", x_raw_key)
                results[i] = hit  # already moved to the right device by cache.get
            else:
                miss_ids.append(i)
                miss_keys.append(key)

        # If anything missed the cache, run the model on those items only
        if len(miss_ids) > 0:
            # Keep the original code's convention of passing a list of per-sample tensors
            inputs = self._prepare_inputs([x[i] for i in miss_ids])
            outputs = self.model(**inputs, output_hidden_states=output_hidden_states)

            # Split tokens for ViT backbones (CLS + registers + patch tokens),
            # otherwise return the spatial map for ConvNeXt.
            model_type = getattr(self.model.config, "model_type", "")
            if "dinov3_vit" in model_type:
                hidden = outputs.last_hidden_state  # [Bm, 1+R+Npatch, D]
                num_reg = getattr(self.model.config, "num_register_tokens", 0)
                patch_tokens = hidden[:, 1 + num_reg :, :]  # [Bm, Npatch, D]
                feats_miss = rearrange(patch_tokens, "b (h w) c -> b c h w", h=sH, w=sW)
            else:
                # ConvNeXt (or similar): already [Bm, C, H', W']
                feats_miss = outputs.last_hidden_state  # [Bm, C, sH, sW] expected

            # Write results back in the original order and populate the cache (CPU copy)
            for j, idx in enumerate(miss_ids):
                feat = feats_miss[j]
                results[idx] = feat
                if getattr(self, "_cache", None) is not None:
                    self._cache.put(miss_keys[j], feat.detach().cpu())

        # Stack all per-sample features into a batch
        features = torch.stack(results, dim=0)  # [B, C, sH, sW]
        return features

    # Convenience to introspect cache if enabled
    def cache_stats(self):
        return None if self._cache is None else self._cache.stats()

# --------- Example usage ----------
if __name__ == "__main__":
    # Example with a BCHW tensor in [-1, 1]
    B, C, H, W = 2, 3, 256, 256
    x = torch.rand(B, C, H, W) * 2.0 - 1.0  # [-1, 1]

    wrapper = Dinov3Wrapper(
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        input_range=(-1.0, 1.0),
        # dtype=torch.bfloat16,
        enable_cache=True,            # << turn on two-tier LRU
        mem_cache_in_gb=16, # 32GB RAM budget
        disk_cache_dir=None,          # default ~/.cache/dinov3_feature_cache
        disk_cache_bytes=None,        # unlimited disk (set a number to cap)
        cache_hidden_states=False,    # skip hidden_states to save space
    )
    wrapper = wrapper.to("cuda:0")

    feats = wrapper.get_features(x, x_raw_keys=["batch-0001", "batch-0002"], output_hidden_states=False)
    print("Patch/Spatial:", feats.shape)

    feats = wrapper.get_features(x, x_raw_keys=["batch-0001", "batch-0002"], output_hidden_states=False)

    # Cache stats
    print("Cache:", wrapper.cache_stats())
