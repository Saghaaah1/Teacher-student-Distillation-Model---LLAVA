"""
StudentLLaVA — Part 1: Shallow Model Design
============================================

    Image → [CLIP ViT-L/14-336] → Z_v → [Projection MLP] → H_v ──┐
                                                                    ├──► [TinyLlama-1.1B] → Response
    Text Instruction ────────────────────────────────────── H_q ───┘
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig, AutoModelForCausalLM


TEACHER_PATH   = "/home/idekube/models/swift/llava-1___5-7b-hf"
TINYLLAMA_PATH = "/home/idekube/models/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0"

VISION_DIM = 1024
LLM_DIM    = 2048


# ---------------------------------------------------------------------------
# Projection MLP
# ---------------------------------------------------------------------------

class VisualProjector(nn.Module):
    def __init__(self, vision_dim: int = VISION_DIM, llm_dim: int = LLM_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Helper: extract CLIP weights from LLaVA checkpoint
# ---------------------------------------------------------------------------

def load_clip_from_llava(llava_path: str) -> CLIPVisionModel:
    """
    In the LLaVA checkpoint, CLIP weights are stored under the prefix
    'vision_tower.vision_model.*'. We strip that prefix and load them
    directly into a CLIPVisionModel.
    """
    import json, os
    from safetensors.torch import load_file

    # Build CLIP model from config
    with open(os.path.join(llava_path, "config.json")) as f:
        full_cfg = json.load(f)
    vision_cfg = CLIPVisionConfig(**full_cfg["vision_config"])
    clip_model = CLIPVisionModel(vision_cfg)

    # Load all shards and filter CLIP weights
    index_path = os.path.join(llava_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Find which shards contain vision_tower weights
    shards_needed = set()
    for key, shard in index["weight_map"].items():
        if key.startswith("vision_tower.vision_model."):
            shards_needed.add(shard)

    # Load those shards and remap keys
    clip_state = {}
    for shard in shards_needed:
        shard_path = os.path.join(llava_path, shard)
        weights = load_file(shard_path)
        for key, val in weights.items():
            if key.startswith("vision_tower.vision_model."):
                new_key = key.replace("vision_tower.", "")  # -> vision_model.*
                clip_state[new_key] = val

    missing, unexpected = clip_model.load_state_dict(clip_state, strict=True)
    print(f"[CLIP] Loaded from LLaVA. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return clip_model


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

class StudentLLaVA(nn.Module):

    def __init__(
        self,
        clip_path: str = TEACHER_PATH,
        llm_path: str  = TINYLLAMA_PATH,
        freeze_vision: bool = True,
        freeze_llm: bool    = True,
    ):
        super().__init__()

        # Vision encoder — extracted from LLaVA checkpoint
        print(f"[Student] Loading CLIP from: {clip_path}")
        self.vision_encoder = load_clip_from_llava(clip_path)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            print("[Student] Vision encoder frozen.")

        # Projection MLP
        self.projector = VisualProjector(VISION_DIM, LLM_DIM)

        # LLM — TinyLlama
        print(f"[Student] Loading LLM from: {llm_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            dtype=torch.float16,
        )
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            print("[Student] LLM frozen (Stage 1).")
        else:
            print("[Student] LLM trainable (Stage 2).")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Uses layer -2 hidden state, as specified in teacher config.json:
            vision_feature_layer = -2
        Removes [CLS] token, keeps 576 patch tokens.
        """
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        visual_features = outputs.hidden_states[-2][:, 1:, :]  # (B, 576, 1024)

        # Cast to projector dtype (float32) before MLP
        visual_features = visual_features.to(self.projector.proj[0].weight.dtype)
        return self.projector(visual_features)                  # (B, 576, 2048)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        visual_tokens = self.encode_image(pixel_values)
        visual_mask   = torch.ones(
            visual_tokens.shape[:2],
            dtype=torch.long,
            device=visual_tokens.device,
        )
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        mask   = torch.cat([visual_mask,   attention_mask], dim=1)

        return self.llm(
            inputs_embeds=embeds.to(self.llm.dtype),
            attention_mask=mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

    def set_stage2(self):
        for p in self.llm.parameters():
            p.requires_grad = True
        print("[Student] Stage 2 — LLM unfrozen.")

    def print_param_summary(self):
        print(f"\n{'='*52}")
        print(f"  StudentLLaVA — Parameter Summary")
        print(f"{'='*52}")
        for name, module in [
            ("CLIP ViT-L/14-336 (frozen)", self.vision_encoder),
            ("Projector 2-layer MLP",      self.projector),
            ("TinyLlama-1.1B",             self.llm),
        ]:
            total     = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name:<30} {total/1e6:7.1f}M  |  {trainable/1e6:6.1f}M trainable")
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{'─'*52}")
        print(f"  {'TOTAL':<30} {total/1e6:7.1f}M  |  {trainable/1e6:6.1f}M trainable")
        print(f"{'='*52}\n")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = StudentLLaVA(freeze_vision=True, freeze_llm=True)
    model.print_param_summary()

    B = 1
    pixel_values = torch.randn(B, 3, 336, 336)
    input_ids    = torch.randint(0, 32000, (B, 10))
    attn_mask    = torch.ones(B, 10, dtype=torch.long)

    with torch.no_grad():
        out = model(pixel_values, input_ids, attn_mask)

    print(f"Logits shape        : {out.logits.shape}")
    print(f"Hidden states count : {len(out.hidden_states)}")