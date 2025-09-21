import json
import requests
import random
from danbooru_api.main import main
import re
import os
import time
from predict import select_best_4level_flat
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Функция для получения тегов по дате (рекурсивно)
def get_tags_for_date(current_date, depth=0, max_depth=30):
    if depth > max_depth:
        print("Достигнут максимум откатов по датам (30 дней). Нет тегов.")
        return []
    
    date_str = current_date.strftime('%Y-%m-%d')
    url = f'https://danbooru.donmai.us/explore/posts/searches?date={date_str}'
    
    # Настройка прокси (закомментируйте, если не работает)
    proxy = "http://npyuqomx:jpod2zw7iwg1@84.247.60.125:6095"
    proxies = {
        'http': proxy,
        'https': proxy
    }
    
    # Создаём сессию
    session = requests.Session()
    session.proxies.update(proxies)  # Закомментируйте, если прокси не работает
    
    # Retry-логика
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            break
        except requests.exceptions.ConnectionError as e:
            print(f"Ошибка соединения (попытка {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Все попытки исчерпаны.")
                return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth)
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"Другая ошибка запроса: {e}")
            return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth)
    
    if not response:
        return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth)
    
    # Парсинг HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Поиск таблицы с тегами
    table = soup.find('tbody')
    if not table:
        print("Таблица с тегами не найдена.")
        return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth)
    
    # Извлечение строк
    rows = table.find_all('tr')
    data = []
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            tag_cell = cells[0].find('a')
            tag = tag_cell.text.strip() if tag_cell else 'N/A'
            count_str = cells[1].text.strip()
            try:
                count_val = int(count_str)
            except ValueError:
                count_val = 0
            data.append({'tag': tag, 'count': count_val})
    
    # Сортировка по count (убывание)
    data = sorted(data, key=lambda x: x['count'], reverse=True)
    
    if not data:
        print(f"Тегов не найдено для {date_str}, откатываемся на предыдущий день.")
        return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth)
    
    return data

# Читаем tags.txt
txt_file = r'D:\Python Scripts\ai\danbooru_api\character_tags.txt'
try:
    with open(txt_file, 'r', encoding='utf-8') as f:
        existing_tags = set(line.strip() for line in f if line.strip())
except FileNotFoundError:
    existing_tags = set()
    print(f"Файл {txt_file} не найден, создаём пустой.")
    with open(txt_file, 'w', encoding='utf-8') as f:
        pass

# Set для использованных тегов
used_tags = set()

count = 3
search_type = "character"  # character/general
tags = [["top character", "character"], ["top character", "character"], ["top character", "character"]] # top character/random character
rating = "all"  # questionable/sensitive/general/all

# Цикл на count раз
for i in range(0, count):
    if isinstance(tags, list):
        tag = tags[i][0]
        search_type = tags[i][1]
    if tag == "top character":
        today = datetime.now()
        
        # Получаем теги (рекурсивно, если нужно)
        tags_data = get_tags_for_date(today)
        
        if not tags_data:
            print(f"Итерация {i}: Не удалось получить теги даже после откатов.")
            continue
        
        # Ищем первый тег из топа, который есть в txt и не использован
        for item in tags_data:
            tag_candidate = item['tag']
            if tag_candidate in existing_tags and tag_candidate not in used_tags:
                tag = tag_candidate
                used_tags.add(tag_candidate)
                break
    
    # Используем выбранный тег вместо 'random character'
    posts, user_tag_formatted = main(search_type, tag, rating)

    parts = re.split(r'(\(.*?\))', user_tag_formatted.split(" |")[0])
    hashtags = []

    for part in parts:
        if not part.strip():
            continue
        clean = part.replace('(', '').replace(')', '').replace(' ', '').replace(':', '')
        hashtags.append(f"#{clean}")

    description = f"{user_tag_formatted}\n\nThe first buyer on DeviantArt also receives an archive containing the full set of 125 images in the highest quality.\n\nYou can also check out my Patreon, which includes all the sets plus a variety of other exclusive content only for Patreon subscribers.\n\n{' '.join(hashtags)}"

    API_URL = "http://127.0.0.1:8188/prompt"

    copy_from_dir = r"D:\StabilityMatrix\Data\Packages\ComfyUI\output"
    expected_count = 125
    check_interval = 60

    # Путь к файлу workflow
    workflow_file = "Unsaved Workflow(1).json"

    # Чтение workflow из файла
    with open(workflow_file, "r") as f:
        workflow = json.load(f)

    # Новый промпт
    base_prompt = ", (gogalking:0.8), (loika:0.15), (33 gaff:0.05), masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, detailed eyes"

    for post in posts:
        new_prompt = post["tags"]
        full_prompt = new_prompt + base_prompt

        if "110" in workflow:
            workflow["110"]["inputs"]["positive"] = full_prompt

        # Запоминаем, сколько файлов было изначально
        prev_count = len([f for f in os.listdir(copy_from_dir) 
                         if os.path.isfile(os.path.join(copy_from_dir, f)) and f.lower().endswith('.png')])

        for j in range(1, 6):
            random_seed = random.getrandbits(64)
            workflow["589"]["inputs"]["seed"] = random_seed

            data = {
                "client_id": 1,
                "prompt": workflow
            }

            # Отправка workflow на выполнение
            response = requests.post(API_URL, json=data)

            if response.status_code != 200:
                print(f"Ошибка: {response.status_code}")
                print(response.text)

        while True:
            png_files = [f for f in os.listdir(copy_from_dir) 
                        if os.path.isfile(os.path.join(copy_from_dir, f)) and f.lower().endswith('.png')]
            current_count = len(png_files)

            if current_count - prev_count >= 5:
                prev_count = current_count
                break

            time.sleep(10)

    while True:
        png_files = [f for f in os.listdir(copy_from_dir) if os.path.isfile(os.path.join(copy_from_dir, f)) and f.lower().endswith('.png')]
        
        if len(png_files) >= expected_count:
            input_dir = r"D:\StabilityMatrix\Data\Packages\ComfyUI\output\dataset"
            model_path = "7293 4592 best.pth"
            output_dir = r"D:\finish"

            select_best_4level_flat(
                model_path=model_path,
                input_dir=input_dir,
                group_size=5,
                batch_size=125,
                save_threshold=2.0,
                output_dir=output_dir,
                copy_from_dir=copy_from_dir,
                description=description
            )
            break
        
        time.sleep(check_interval)

os.system("shutdown /s /t 60")