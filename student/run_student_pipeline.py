import torch
from transformers import AutoTokenizer

from student_model import StudentMLLM


VISION_MODEL_PATH = "./shared/vision_encoder/siglip-so400m-patch14-384"
STUDENT_LLM_PATH = "./student/qwen2.5-0.5b"


def load_student_and_tokenizer():
    model = StudentMLLM(
        vision_model_path=VISION_MODEL_PATH,
        llm_model_path=STUDENT_LLM_PATH,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_LLM_PATH)
    return model, tokenizer


def build_dummy_inputs(prompt: str = "Describe.", image_size: int = 384):
    dummy_image = torch.randn(1, 3, image_size, image_size)
    return prompt, dummy_image


def get_visual_tokens(model: StudentMLLM, image: torch.Tensor, max_visual_tokens: int | None = None):
    with torch.no_grad():
        vision_outputs = model.vision_encoder(pixel_values=image)
        vision_tokens = vision_outputs.last_hidden_state

        if max_visual_tokens is not None:
            vision_tokens = vision_tokens[:, :max_visual_tokens, :]

        projected_tokens = model.projector(vision_tokens)

    return vision_tokens, projected_tokens


def get_text_embeddings(model: StudentMLLM, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        text_embeds = model.llm.get_input_embeddings()(input_ids)

    return input_ids, attention_mask, text_embeds


def fuse_modalities(projected_tokens: torch.Tensor, text_embeds: torch.Tensor, text_attention_mask: torch.Tensor):
    multimodal_embeds = torch.cat([projected_tokens, text_embeds], dim=1)

    visual_attention = torch.ones(projected_tokens.shape[:2], dtype=torch.long)
    multimodal_attention_mask = torch.cat([visual_attention, text_attention_mask], dim=1)

    return multimodal_embeds, multimodal_attention_mask


def forward_llm(model: StudentMLLM, multimodal_embeds: torch.Tensor, multimodal_attention_mask: torch.Tensor):
    llm_dtype = model.llm.get_input_embeddings().weight.dtype
    multimodal_embeds = multimodal_embeds.to(llm_dtype)

    with torch.no_grad():
        outputs = model.llm(
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_attention_mask,
        )

    return outputs, llm_dtype, multimodal_embeds


def decode_next_token(outputs, tokenizer):
    last_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(last_token_logits, dim=-1)
    decoded = tokenizer.decode(next_token_id[0])
    return next_token_id.item(), decoded


def run_pipeline(prompt: str = "Describe.", max_visual_tokens: int = 10):
    model, tokenizer = load_student_and_tokenizer()
    _, dummy_image = build_dummy_inputs(prompt=prompt)

    vision_tokens, projected_tokens = get_visual_tokens(
        model=model,
        image=dummy_image,
        max_visual_tokens=max_visual_tokens,
    )

    input_ids, text_attention_mask, text_embeds = get_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    multimodal_embeds, multimodal_attention_mask = fuse_modalities(
        projected_tokens=projected_tokens,
        text_embeds=text_embeds,
        text_attention_mask=text_attention_mask,
    )

    outputs, llm_dtype, multimodal_embeds = forward_llm(
        model=model,
        multimodal_embeds=multimodal_embeds,
        multimodal_attention_mask=multimodal_attention_mask,
    )

    next_token_id, decoded = decode_next_token(outputs, tokenizer)

    print("=== Student Multimodal Pipeline ===")
    print(f"Prompt: {prompt}")
    print(f"Vision tokens shape: {vision_tokens.shape}")
    print(f"Projected visual tokens shape: {projected_tokens.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Text embeddings shape: {text_embeds.shape}")
    print(f"Multimodal embeddings shape: {multimodal_embeds.shape}")
    print(f"Multimodal attention mask shape: {multimodal_attention_mask.shape}")
    print(f"Attention mask sum: {multimodal_attention_mask.sum().item()}")
    print(f"LLM dtype: {llm_dtype}")
    print(f"Multimodal embeds dtype: {multimodal_embeds.dtype}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Next token id: {next_token_id}")
    print(f"Decoded token: {repr(decoded)}")


if __name__ == "__main__":
    run_pipeline(prompt="Describe.", max_visual_tokens=10)