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
– Hair (e.g., swaying, fluttering gently in an unseen breeze)
– Clothing or fabric elements (e.g., ribbons, capes, loose cloth reacting to subtle air movement)
– Ambient particles (e.g., dust motes, floating embers, drifting petals)
– Light effects (e.g., gentle pulsing glows, soft energy flickers, holographic shimmer)
– Floating objects (e.g., orbs, small drones) if they are clearly not rigidly fixed
– Background **ambient** motion (e.g., slow drifting fog, gentle parallax of distant layers, faint cloud movement)
🚫 And **explicitly specify what should remain completely static**:
– Face and all facial features — no mimicry, no blinking, no eye movement, no lip movement, no micro-expressions whatsoever
– Rigid structures (e.g., weapons, armor, furniture, architecture)
– Body parts not involved in very subtle idle motion (e.g., torso, arms, legs — only extremely gentle breathing if clearly suggested)
– Background elements that do not visually suggest any motion

Important rule that must be clearly stated in every generated description:
«Mimicry and facial animation are not needed and should be completely absent — face remains perfectly still and expressionless throughout the entire loop.»

⚠️ Strict guidelines:
– Animation must be fluid, consistent, perfectly seamless and loopable
– NO facial animation of any kind is allowed
– NO sudden motions, no pose changes, no transitions
– Do NOT add or invent any objects/effects not visibly present in the image
– Do NOT describe colors, character names, story themes or static appearance details
– Return only the pure animation description text — no lists, no markdown, no extra commentary"""
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