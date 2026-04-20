import torch
import torch.nn as nn
from transformers import SiglipVisionModel, AutoModelForCausalLM


class VisualProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.proj(x)


class StudentMLLM(nn.Module):
    def __init__(self, vision_model_path: str, llm_model_path: str):
        super().__init__()

        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path)

        vision_dim = 1152
        llm_dim = self.llm.config.hidden_size

        self.projector = VisualProjector(vision_dim=vision_dim, llm_dim=llm_dim)

    def forward(self, pixel_values):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_tokens = vision_outputs.last_hidden_state
        projected_tokens = self.projector(vision_tokens)
        return projected_tokens


if __name__ == "__main__":
    model = StudentMLLM(
        vision_model_path="./shared/vision_encoder/siglip-so400m-patch14-384",
        llm_model_path="./student/qwen2.5-0.5b"
    )

    print("Student model loaded successfully.")
    print("LLM hidden size:", model.llm.config.hidden_size)
    print("Projector:", model.projector)
