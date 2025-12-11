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
from vlm import generate_animation_prompt
import shutil
import tempfile
import subprocess
from PIL import Image

def get_gif_color_count(path):
    """Определение количества уникальных цветов в первом кадре GIF."""
    img = Image.open(path)
    img = img.convert("RGB")
    colors = img.getcolors(maxcolors=3000000)
    if colors is None:
        return 256
    return len(colors)

def create_compressed_gif(input_path, max_size_mb=15):
    max_bytes = max_size_mb * 1024 * 1024

    base, ext = os.path.splitext(input_path)
    output_path = base + "_compressed" + ext

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "compressed.gif")

    # Определяем максимальное количество цветов
    original_color_count = get_gif_color_count(input_path)
    print(f"[INFO] Colors in GIF: {original_color_count}")

    # 1. Если уже меньше лимита → просто копируем
    if os.path.getsize(input_path) <= max_bytes:
        shutil.copy(input_path, output_path)
        shutil.rmtree(temp_dir)
        return output_path

    # 2. Оптимизация без потерь O1-O3
    for level in [1, 2, 3]:
        subprocess.run([
            r"C:\ProgramData\chocolatey\bin\gifsicle.exe",
            f"-O{level}",
            "--colors", str(original_color_count),
            input_path,
            "-o", temp_path
        ])
        if os.path.exists(temp_path) and os.path.getsize(temp_path) <= max_bytes:
            shutil.move(temp_path, output_path)
            shutil.rmtree(temp_dir)
            return output_path

    # 3. Плавное уменьшение количества цветов от исходного
    color_steps = [
        int(original_color_count * 0.9),
        int(original_color_count * 0.75),
        int(original_color_count * 0.6),
        int(original_color_count * 0.5),
        int(original_color_count * 0.4),
        int(original_color_count * 0.33),
        128, 100, 80, 64, 48, 32
    ]

    # Убираем повторяющиеся / нулевые / слишком высокие значения
    color_steps = sorted({c for c in color_steps if 1 < c <= original_color_count}, reverse=True)

    for colors in color_steps:
        subprocess.run([
            r"C:\ProgramData\chocolatey\bin\gifsicle.exe",
            "-O3",
            "--colors", str(colors),
            input_path,
            "-o", temp_path
        ])
        if os.path.getsize(temp_path) <= max_bytes:
            shutil.move(temp_path, output_path)
            shutil.rmtree(temp_dir)
            return output_path

    # 4. В крайнем случае — немного lossy
    subprocess.run([
        r"C:\ProgramData\chocolatey\bin\gifsicle.exe",
        "-O3",
        "--lossy=20",
        "--delay=4",
        input_path,
        "-o", temp_path
    ])

    shutil.move(temp_path, output_path)
    shutil.rmtree(temp_dir)
    return output_path

def safe_move(src, dst, retries=20, delay=1):
    """Перемещает файл, ожидая пока он разблокируется."""
    for attempt in range(retries):
        try:
            shutil.move(src, dst)
            return True
        except PermissionError:
            print(f"[INFO] Файл занят, повтор попытки {attempt + 1}/{retries}...")
            time.sleep(delay)
    print("[ERROR] Не удалось переместить файл — он постоянно занят!")
    return False

def wait_until_finished(path):
    last_size = -1
    while True:
        try:
            size = os.path.getsize(path)
            if size == last_size:
                return
            last_size = size
        except:
            pass
        time.sleep(1)

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
tags = [["ningguang_(genshin_impact)", "character"], ["raiden_shogun", "character"], ["saber_(fate)", "character"]]
rating = "general"

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

    copy_from_dir = r"D:\StabilityMatrix\Data\Packages\ComfyUIfix\output"
    expected_count = 125
    check_interval = 60

    workflow_file = "base.json"

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

    urls_for_anim = []

    while True:
        png_files = [f for f in os.listdir(copy_from_dir) if os.path.isfile(os.path.join(copy_from_dir, f)) and f.lower().endswith('.png')]
        
        if len(png_files) >= expected_count:
            input_dir = r"D:\StabilityMatrix\Data\Packages\ComfyUIfix\output\dataset"
            model_path = "7186 4182 6364 best.pth"
            output_dir = r"D:\finish"

            urls_for_anim.append(select_best_4level_flat(
                model_path=model_path,
                input_dir=input_dir,
                group_size=5,
                batch_size=125,
                save_threshold=2.0,
                output_dir=output_dir,
                copy_from_dir=copy_from_dir,
                description=description
            ))
            break
        
        time.sleep(check_interval)

    for url in urls_for_anim:
        prompt = generate_animation_prompt(url)

        workflow_file = "ez.json"

        with open(workflow_file, "r") as f:
            workflow = json.load(f)

        workflow["17"]["inputs"]["image"] = url

        random_seed = random.getrandbits(64)
        workflow["13"]["inputs"]["noise_seed"] = random_seed

        workflow["15"]["inputs"]["text"] = prompt

        data = {
            "client_id": 1,
            "prompt": workflow
        }

        response = requests.post(API_URL, json=data)

        found_gif = None

        while True:
            for root, dirs, files in os.walk(copy_from_dir):
                for file in files:
                    if file.lower().endswith(".gif"):
                        found_gif = os.path.join(root, file)
                        break
                if found_gif:
                    break

            if found_gif:
                break

            time.sleep(10)
        
        target_dir = os.path.dirname(url)
        gif_target_path = os.path.join(target_dir, os.path.basename(found_gif))

        wait_until_finished(found_gif)
        safe_move(found_gif, gif_target_path)

        if os.path.exists(url):
            os.remove(url)

        txt_path = os.path.splitext(url)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(prompt)

        for root, dirs, files in os.walk(copy_from_dir):
            for file in files:
                if file.lower().endswith((".gif", ".png")):
                    try:
                        os.remove(os.path.join(root, file))
                    except:
                        pass

        create_compressed_gif(gif_target_path, 15)

os.system("shutdown /s /t 60")