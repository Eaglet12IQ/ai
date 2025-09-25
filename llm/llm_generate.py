import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle
import os

# Функция для генерации уникального промпта
def generate_unique_prompt(input_word, model, tokenizer, all_tags_set, max_length=200):
    model.eval()
    input_text = f"generate prompt: {input_word}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=20,  # Ещё больше уменьшаем для выбора только самых вероятных тегов
        top_p=0.5,  # Ограничиваем выбор наиболее вероятными тегами
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,  # Увеличиваем штраф за повторы
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tags = [tag.strip() for tag in generated.split(',')]
    valid_tags = [tag for tag in tags if tag in all_tags_set]
    return ', '.join(valid_tags)

def main():
    # Проверка наличия файлов
    model_dir = 'llm/t5_prompt_model'
    tags_file = 'llm/all_tags.pkl'
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please ensure the model is trained and saved.")
    if not os.path.exists(tags_file):
        raise FileNotFoundError(f"Tags file '{tags_file}' not found. Please ensure it exists.")

    # Загрузка тегов
    with open(tags_file, 'rb') as f:
        all_tags = pickle.load(f)
    all_tags_set = set(all_tags)

    # Загрузка модели и токенизатора
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
    except Exception as e:
        raise Exception(f"Error loading model or tokenizer: {e}")

    # Перенос модели на доступное устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Список тестовых слов
    test_words = ['hatsune miku, ravel phenex']

    # Генерация промптов
    for word in test_words:
        print(f"\nВходное слово: {word}")
        for i in range(3):  # Генерируем 3 варианта для каждого слова
            prompt = generate_unique_prompt(word, model, tokenizer, all_tags_set)
            print(f"Сгенерированный промпт {i+1}: {prompt}")

if __name__ == "__main__":
    main()