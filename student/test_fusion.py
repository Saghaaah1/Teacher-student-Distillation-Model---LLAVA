import torch
from transformers import AutoTokenizer
from student_model import StudentMLLM

model = StudentMLLM(
    vision_model_path="./shared/vision_encoder/siglip-so400m-patch14-384",
    llm_model_path="./student/qwen2.5-0.5b"
)

model.eval()
tokenizer = AutoTokenizer.from_pretrained("./student/qwen2.5-0.5b")

prompt = "Describe the image."
inputs = tokenizer(prompt, return_tensors="pt")

dummy_image = torch.randn(1, 3, 384, 384)

with torch.no_grad():
    vision_outputs = model.vision_encoder(pixel_values=dummy_image)
    vision_tokens = vision_outputs.last_hidden_state
    projected_tokens = model.projector(vision_tokens)

    input_ids = inputs["input_ids"]
    text_embeds = model.llm.get_input_embeddings()(input_ids)

    multimodal_embeds = torch.cat([projected_tokens, text_embeds], dim=1)

print("Projected visual tokens shape:", projected_tokens.shape)
print("Text embeddings shape:", text_embeds.shape)
print("Multimodal embeddings shape:", multimodal_embeds.shape)
