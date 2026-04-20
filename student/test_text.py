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

with torch.no_grad():
    input_ids = inputs["input_ids"]
    text_embeds = model.llm.get_input_embeddings()(input_ids)

print("Input IDs shape:", input_ids.shape)
print("Text embeddings shape:", text_embeds.shape)
print("Student hidden size:", model.llm.config.hidden_size)
