import torch
from student_model import StudentMLLM

model = StudentMLLM(
    vision_model_path="./shared/vision_encoder/siglip-so400m-patch14-384",
    llm_model_path="./student/qwen2.5-0.5b"
)

model.eval()

dummy_image = torch.randn(1, 3, 384, 384)

with torch.no_grad():
    vision_outputs = model.vision_encoder(pixel_values=dummy_image)
    vision_tokens = vision_outputs.last_hidden_state
    projected_tokens = model.projector(vision_tokens)

print("Vision tokens shape:", vision_tokens.shape)
print("Projected tokens shape:", projected_tokens.shape)
print("Student hidden size:", model.llm.config.hidden_size)
