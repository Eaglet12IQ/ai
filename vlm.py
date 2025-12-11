def generate_animation_prompt(image_path: str):
    import torch
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        AutoModelForVision2Seq,
        AutoModelForCausalLM
    )
    from PIL import Image
    import gc

    # ==== МОДЕЛИ ====
    CAPTION_MODEL_ID = "Minthy/ToriiGate-v0.4-7B"
    TEXT_MODEL_ID = "Qwen/Qwen3-8B"

    # ==== 4bit квантование ====
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ================================
    # ========== ЭТАП 1: КАПШЕН ==========
    # ================================
    processor_cap = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
    model_cap = AutoModelForVision2Seq.from_pretrained(
        CAPTION_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    image = Image.open(image_path).convert("RGB")

    messages_cap = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text":
             "Describe this image in extreme detail in English. Include every character, pose, clothing, expression, lighting, colors, atmosphere, background objects, and tiny details. Write 7–10 long, packed sentences."
            }
        ]}
    ]

    prompt_cap = processor_cap.apply_chat_template(messages_cap, tokenize=False, add_generation_prompt=True)
    inputs_cap = processor_cap(text=[prompt_cap], images=[[image]], return_tensors="pt").to("cuda")

    generated = model_cap.generate(
        **inputs_cap,
        max_new_tokens=550,
        temperature=0.2,
    )
    caption = processor_cap.decode(generated[0], skip_special_tokens=True).split("assistant")[-1].strip()

    # Очистка VRAM
    del model_cap, processor_cap, inputs_cap, generated
    gc.collect()
    torch.cuda.empty_cache()

    # ================================
    # ========== ЭТАП 2: QWEN ==========
    # ================================
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model_text = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    system_prompt = (
        """You are an expert in motion design for seamless animated loops.

Given a single image as input, generate a richly detailed description of how it could be turned into a smooth, seamless animation.

Your response must include:

✅ What elements **should move**:
– Hair (e.g., swaying, fluttering)
– Eyes (e.g., blinking, subtle gaze shifts)
– Clothing or fabric elements (e.g., ribbons, loose parts reacting to wind or motion)
– Ambient particles (e.g., dust, sparks, petals)
– Light effects (e.g., holograms, glows, energy fields)
– Floating objects (e.g., drones, magical orbs) if they are clearly not rigid or fixed
– Background **ambient** motion (e.g., fog, drifting light, slow parallax)

🚫 And **explicitly specify what should remain static**:
– Rigid structures (e.g., chairs, weapons, metallic armor)
– Body parts not involved in subtle motion (e.g., torso, limbs unless there’s idle shifting)
– Background elements that do not visually suggest movement

⚠️ Guidelines:
– The animation must be **fluid, consistent, and seamless**, suitable for a loop  
– Do NOT include sudden movements, teleportation, scene transitions, or pose changes  
– Do NOT invent objects or effects not present in the image  
– Do NOT describe static features like colors, names, or environment themes  
– Return only the description (no lists, no markdown, no instructions)"""
    )

    user_prompt = f"""Detailed image description:
\"\"\"{caption}\"\"\"

Create the final animation prompt. Follow the rules exactly."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    ).to("cuda")

    output = model_text.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1500,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # Убираем возможный служебный <think> от Qwen3
    if "</think>" in result:
        result = result.split("</think>", 1)[1]

    final_prompt = result.strip('“”"\'').strip()

    # Очистка VRAM
    del model_text, tokenizer, inputs, output
    gc.collect()
    torch.cuda.empty_cache()

    return final_prompt