import os
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Шлях до файн-тюненої моделі можна перевизначити через змінну оточення
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/finetuned")
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

app = FastAPI(title="Finetuned Qwen API")

# Завантаження моделі та токенайзера при старті сервера
@app.on_event("startup")
def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()


class GenerationRequest(BaseModel):
    instruction: str
    input: Optional[str] = ""
    max_new_tokens: int = 128


class GenerationResponse(BaseModel):
    output: str


@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    if req.input and req.input.strip():
        prompt = f"Instruction: {req.instruction}\nInput: {req.input}\nOutput:"
    else:
        prompt = f"Instruction: {req.instruction}\nOutput:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Output:" in generated_text:
        generated_text = generated_text.split("Output:")[-1].strip()
    else:
        generated_text = generated_text.replace(prompt, "").strip()

    return GenerationResponse(output=generated_text)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
