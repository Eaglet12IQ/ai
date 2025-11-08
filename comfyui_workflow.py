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

def load_proxies_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        proxies = []
        for line in lines:
            if line.strip():
                parts = line.strip().split(':')
                if len(parts) == 4:
                    ip, port, username, password = parts
                    proxies.append(f"http://{username}:{password}@{ip}:{port}")
        return proxies
    except requests.RequestException as e:
        print(f"Ошибка при загрузке списка прокси: {e}")
        return []

def get_tags_for_date(current_date, depth=0, max_depth=30, proxies_list=None):
    if depth > max_depth:
        print("Достигнут максимум откатов по датам (30 дней). Нет тегов.")
        return []
    
    date_str = current_date.strftime('%Y-%m-%d')
    url = f'https://danbooru.donmai.us/explore/posts/searches?date={date_str}'
    
    if not proxies_list:
        proxies_list = load_proxies_from_url("https://proxy.webshare.io/api/v2/proxy/list/download/jxerjrnkysbdnhlzhnhnglewhvjalpupcunqxutc/-/any/username/direct/-/?plan_id=11389346")
    
    session = requests.Session()
    
    if not proxies_list:
        print("Ошибка: список прокси пуст. Выполняю запрос без прокси.")
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            print(f"Ошибка соединения без прокси: {e}")
            return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth, proxies_list)
        except requests.exceptions.RequestException as e:
            print(f"Другая ошибка запроса без прокси: {e}")
            return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth, proxies_list)
    else:
        max_retries = 3
        for proxy in proxies_list:
            session.proxies.update({'http': proxy, 'https': proxy})
            for attempt in range(max_retries):
                try:
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
                    break
                except requests.exceptions.ConnectionError as e:
                    print(f"Ошибка соединения с прокси {proxy} (попытка {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        print(f"Все попытки с прокси {proxy} исчерпаны, пробую следующий.")
                        break
                    time.sleep(2 ** attempt)
                except requests.exceptions.RequestException as e:
                    print(f"Другая ошибка запроса с прокси {proxy}: {e}")
                    break
            else:
                continue
            break
        else:
            print("Ошибка: все прокси из списка не работают.")
            return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth, proxies_list)
    
    if not response:
        return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth, proxies_list)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    table = soup.find('tbody')
    if not table:
        print("Таблица с тегами не найдена.")
        return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth, proxies_list)
    
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
    
    data = sorted(data, key=lambda x: x['count'], reverse=True)
    
    if not data:
        print(f"Тегов не найдено для {date_str}, откатываемся на предыдущий день.")
        return get_tags_for_date(current_date - timedelta(days=1), depth + 1, max_depth, proxies_list)
    
    return data

def find_valid_tag(current_date, existing_tags, used_tags, depth=0, max_depth=30):
    if depth > max_depth:
        print("Достигнут максимум рекурсивных вызовов для поиска тега (30 дней).")
        return None
    
    tags_data = get_tags_for_date(current_date)
    if not tags_data:
        print(f"Не удалось получить теги для {current_date.strftime('%Y-%m-%d')}. Пробуем следующий день.")
        return find_valid_tag(current_date - timedelta(days=1), existing_tags, used_tags, depth + 1, max_depth)
    
    for item in tags_data:
        tag_candidate = item['tag']
        if tag_candidate in existing_tags and tag_candidate not in used_tags:
            return tag_candidate
    
    print(f"Не найдено подходящих тегов для {current_date.strftime('%Y-%m-%d')}. Пробуем следующий день.")
    return find_valid_tag(current_date - timedelta(days=1), existing_tags, used_tags, depth + 1, max_depth)

txt_file = r'D:\Python Scripts\ai\danbooru_api\character_tags.txt'
try:
    with open(txt_file, 'r', encoding='utf-8') as f:
        existing_tags = set(line.strip() for line in f if line.strip())
except FileNotFoundError:
    existing_tags = set()
    print(f"Файл {txt_file} не найден, создаём пустой.")
    with open(txt_file, 'w', encoding='utf-8') as f:
        pass

used_tags = set()

count = 3
search_type = "character"
tags = [["kirigaya_suguha", "character"], ["yuuki_(sao)", "character"], ["krista_lenz", "character"]]
rating = "all"

for i in range(0, count):
    if isinstance(tags, list):
        tag = tags[i][0]
        search_type = tags[i][1]
    if tag == "top character":
        today = datetime.now()
        tag = find_valid_tag(today, existing_tags, used_tags)
        if tag is None:
            print(f"Итерация {i}: Не удалось найти подходящий тег после всех попыток.")
            continue
        used_tags.add(tag)
        print(f"Итерация {i}: Выбран тег {tag}")
    
    posts, user_tag_formatted = main(search_type, tag, rating)

    parts = re.split(r'(\(.*?\))', user_tag_formatted.split(" |")[0])
    hashtags = []

    for part in parts:
        if not part.strip():
            continue
        clean = part.replace('(', '').replace(')', '').replace(' ', '').replace(':', '')
        hashtags.append(f"#{clean}")

    description = f"{user_tag_formatted}\n\nYou can also check out my Patreon, which includes all the sets, each containing 125 images.\n\n{' '.join(hashtags)}\n\nThe first buyer of this exclusive also receives an archive containing the full set of 125 images in the highest quality."

    API_URL = "http://127.0.0.1:8188/prompt"

    copy_from_dir = r"D:\StabilityMatrix\Data\Packages\ComfyUI\output"
    expected_count = 125
    check_interval = 60

    workflow_file = "Unsaved Workflow(1).json"

    with open(workflow_file, "r") as f:
        workflow = json.load(f)

    base_prompt = ", (gogalking:0.8), (loika:0.15), (33 gaff:0.05), masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, detailed eyes"

    for post in posts:
        new_prompt = post["tags"]
        full_prompt = new_prompt + base_prompt

        if "110" in workflow:
            workflow["110"]["inputs"]["positive"] = full_prompt

        prev_count = len([f for f in os.listdir(copy_from_dir) 
                         if os.path.isfile(os.path.join(copy_from_dir, f)) and f.lower().endswith('.png')])

        for j in range(1, 6):
            random_seed = random.getrandbits(64)
            workflow["589"]["inputs"]["seed"] = random_seed

            data = {
                "client_id": 1,
                "prompt": workflow
            }

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