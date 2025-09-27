import requests
import sqlite3
import time
from urllib.parse import quote
import re
import random

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

def load_tags_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return {tag.strip() for tag in f.readlines() if tag.strip()}
    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден.")
        return set()

def create_database():
    conn = sqlite3.connect('danbooru_api/danbooru_posts.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS requests
                 (request_id INTEGER PRIMARY KEY AUTOINCREMENT, user_tag_formatted TEXT)''')
    
    c.execute("INSERT OR IGNORE INTO sqlite_sequence (name, seq) VALUES ('requests', 5)")
    c.execute("UPDATE sqlite_sequence SET seq = 5 WHERE name = 'requests'")
    
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='posts'")
    table_exists = c.fetchone()
    
    if table_exists:
        c.execute("PRAGMA table_info(posts)")
        columns = c.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'id' in column_names or 'request_id' not in column_names:
            c.execute('''CREATE TABLE posts_new
                         (post_id INTEGER PRIMARY KEY,
                          tags TEXT,
                          rating_category TEXT,
                          request_id INTEGER,
                          FOREIGN KEY (request_id) REFERENCES requests(request_id))''')
            
            if 'user_tag_formatted' in column_names:
                c.execute("SELECT DISTINCT user_tag_formatted FROM posts")
                unique_tags = c.fetchall()
                tag_to_request_id = {}
                for tag in unique_tags:
                    tag_value = tag[0]
                    c.execute("INSERT INTO requests (user_tag_formatted) VALUES (?)", (tag_value,))
                    tag_to_request_id[tag_value] = c.lastrowid
                c.execute("SELECT seq FROM sqlite_sequence WHERE name = 'requests'")
                current_seq = c.fetchone()[0]
                if current_seq < 1:
                    c.execute("UPDATE sqlite_sequence SET seq = 1 WHERE name = 'requests'")
                
                c.execute('''SELECT post_id, tags, rating_category, user_tag_formatted
                             FROM posts''')
                for row in c.fetchall():
                    post_id, tags, rating_category, user_tag_formatted = row
                    request_id = tag_to_request_id.get(user_tag_formatted, None)
                    if request_id:
                        c.execute('''INSERT INTO posts_new (post_id, tags, rating_category, request_id)
                                     VALUES (?, ?, ?, ?)''', (post_id, tags, rating_category, request_id))
            else:
                c.execute("INSERT INTO requests (user_tag_formatted) VALUES (?)", ("unknown",))
                temp_request_id = c.lastrowid
                c.execute('''INSERT INTO posts_new (post_id, tags, rating_category, request_id)
                             SELECT post_id, tags, rating_category, ? FROM posts''', (temp_request_id,))
            
            c.execute("DROP TABLE posts")
            c.execute("ALTER TABLE posts_new RENAME TO posts")
    else:
        c.execute('''CREATE TABLE posts
                     (post_id INTEGER PRIMARY KEY,
                      tags TEXT,
                      rating_category TEXT,
                      request_id INTEGER,
                      FOREIGN KEY (request_id) REFERENCES requests(request_id))''')
    
    conn.commit()
    conn.close()

def format_tags(character_tags, general_tags, remove_tags):
    character_tag_list = character_tags.split()
    general_tag_list = general_tags.split()
    tags = ' '.join(character_tag_list + general_tag_list).strip()
    if not tags:
        return ""
    tag_list = tags.split()
    filtered_tags = [tag for tag in tag_list if tag not in remove_tags]
    formatted_tags = []
    for tag in filtered_tags:
        if '(' in tag and ')' in tag:
            tag = tag.replace('(', r'\(').replace(')', r'\)')
        tag = tag.replace('_', ' ')
        formatted_tags.append(tag)
    return ', '.join(formatted_tags)

def format_user_tag(user_tag, request_id):
    parts = re.split(r'(\(.*?\))', user_tag)
    formatted_parts = []
    for part in parts:
        if not part.strip():
            continue
        if part.startswith('(') and part.endswith(')'):
            content = part[1:-1].replace('_', ' ')
            words = [word.capitalize() for word in content.split()]
            formatted_parts.append(f"({' '.join(words)})")
        else:
            part = part.replace('_', ' ')
            words = [word.capitalize() for word in part.split()]
            formatted_parts.append(' '.join(words))
    base_result = formatted_parts[0]
    for part in formatted_parts[1:]:
        if part.startswith('('):
            base_result += f" {part}"
    return f"{base_result} | {request_id}th Set"

def fetch_danbooru_posts(tag, page=1, limit=100, proxies_list=None):
    url = f"https://danbooru.donmai.us/posts.json?page={page}&limit={limit}&tags={quote(tag)}"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; DanbooruFetcher/1.0)'}
    auth = ('sunsiutaAI', 'm5vtFYic7ZH4vFM2jZ8gMGYs')
    
    if not proxies_list:
        proxies_list = load_proxies_from_url("https://proxy.webshare.io/api/v2/proxy/list/download/jxerjrnkysbdnhlzhnhnglewhvjalpupcunqxutc/-/any/username/direct/-/?plan_id=11389346")
    
    if not proxies_list:
        print("Ошибка: список прокси пуст. Выполняю запрос без прокси.")
        try:
            response = requests.get(url, headers=headers, auth=auth, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка при запросе без прокси: {e}")
            return []
    
    for proxy in proxies_list:
        proxies = {'http': proxy, 'https': proxy}
        try:
            response = requests.get(url, headers=headers, proxies=proxies, auth=auth, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка с прокси {proxy}: {e}")
            continue
    
    print("Ошибка: все прокси из списка не работают.")
    return []

def get_existing_post_ids():
    conn = sqlite3.connect('danbooru_api/danbooru_posts.db')
    c = conn.cursor()
    c.execute("SELECT post_id FROM posts")
    existing_ids = {row[0] for row in c.fetchall()}
    conn.close()
    return existing_ids

def get_existing_tags_from_db():
    conn = sqlite3.connect('danbooru_api/danbooru_posts.db')
    c = conn.cursor()
    c.execute("SELECT tags FROM posts")
    existing_tags = set()
    for row in c.fetchall():
        if row[0]:
            existing_tags.update(row[0].split(', '))
    conn.close()
    return existing_tags

def save_to_database(new_posts):
    conn = sqlite3.connect('danbooru_api/danbooru_posts.db')
    c = conn.cursor()
    c.executemany(
        "INSERT INTO posts (post_id, tags, rating_category, request_id) VALUES (?, ?, ?, ?)",
        [(post['post_id'], post['tags'], post['rating_category'], post['request_id']) for post in new_posts]
    )
    conn.commit()
    conn.close()

def save_new_tags_to_file(new_tags_set):
    filename = 'danbooru_api/new_tags.txt'
    try:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_tags = {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            existing_tags = set()
        tags_to_add = new_tags_set - existing_tags
        if not tags_to_add:
            return
        with open(filename, 'a', encoding='utf-8') as f:
            for tag in sorted(tags_to_add):
                f.write(f"{tag}\n")
    except Exception as e:
        print(f"Ошибка при сохранении тегов в файл: {e}")

def collect_posts_for_rating(tag, rating, existing_ids, target_count, new_posts, skip_tags, remove_tags, allowed_character_tags, search_type, request_id, existing_tags, new_tags_set):
    page = 1
    search_tag = tag if not rating else f"{tag} rating:{rating}"
    proxies_list = load_proxies_from_url("https://proxy.webshare.io/api/v2/proxy/list/download/jxerjrnkysbdnhlzhnhnglewhvjalpupcunqxutc/-/any/username/direct/-/?plan_id=11389346")
    
    while len(new_posts) < target_count:
        posts = fetch_danbooru_posts(search_tag, page, proxies_list=proxies_list)
        
        if not posts:
            break
        
        for post in posts:
            post_id = post.get('id')
            character_tags = post.get('tag_string_character', '')
            general_tags = post.get('tag_string_general', '')
            all_tags = ' '.join([character_tags, general_tags]).strip()
            tag_set = set(all_tags.split())
            
            if skip_tags & tag_set:
                continue
                
            character_tag_list = character_tags.split()
            flag_allowed_character_tags = True
            for character_tag in character_tag_list:
                if character_tag not in allowed_character_tags:
                    flag_allowed_character_tags = False
                    break
            if not flag_allowed_character_tags:
                continue
                
            tags = format_tags(character_tags, general_tags, remove_tags)
            rating_value = post.get('rating')
            rating_category = 'nsfw' if rating_value in ['q', 's'] else 'sfw'
            
            if post_id and tags and post_id not in existing_ids:
                post_tags = set(tags.split(', '))
                unique_new_tags = post_tags - existing_tags
                if unique_new_tags:
                    new_tags_set.update(unique_new_tags)
                
                new_posts.append({
                    'post_id': post_id,
                    'tags': tags,
                    'rating_category': rating_category,
                    'request_id': request_id
                })
                existing_ids.add(post_id)
                if len(new_posts) >= target_count:
                    break
        
        page += 1
        time.sleep(1)
    
    return new_posts

def main(search_type, tag, rating):
    if search_type not in ['character', 'general']:
        print("Ошибка: неверный тип поиска. Должен быть 'character' или 'general'.")
        return

    target_count = 25
    
    allowed_character_tags = load_tags_from_file('danbooru_api/character_tags.txt')
    if not allowed_character_tags and search_type == 'character':
        print("Ошибка: в файле character_tags.txt не найдены допустимые теги персонажей. Продолжение невозможно.")
        return
    
    if tag == "random character":
        tag = random.choice(list(allowed_character_tags))
    
    if search_type == 'character':
        if tag not in allowed_character_tags:
            print(f"Ошибка: тег '{tag}' не найден в character_tags.txt.")
            return
    
    skip_tags = load_tags_from_file('danbooru_api/skip_tags.txt')
    remove_tags = load_tags_from_file('danbooru_api/remove_tags.txt')

    create_database()
    
    conn = sqlite3.connect('danbooru_api/danbooru_posts.db')
    c = conn.cursor()
    c.execute("INSERT INTO requests (user_tag_formatted) VALUES (?)", ("temp",))
    request_id = c.lastrowid
    user_tag_formatted = format_user_tag(tag, request_id)
    c.execute("UPDATE requests SET user_tag_formatted = ? WHERE request_id = ?", (user_tag_formatted, request_id))
    conn.commit()
    conn.close()
    
    existing_ids = get_existing_post_ids()
    existing_tags = get_existing_tags_from_db()
    new_tags_set = set()
    new_posts = []
    
    valid_ratings = ['questionable', 'sensitive', 'general']
    
    if rating in valid_ratings:
        new_posts = collect_posts_for_rating(tag, rating, existing_ids, target_count, new_posts, skip_tags, remove_tags, allowed_character_tags, search_type, request_id, existing_tags, new_tags_set)
    else:
        for rating in valid_ratings:
            new_posts = collect_posts_for_rating(tag, rating, existing_ids, target_count, new_posts, skip_tags, remove_tags, allowed_character_tags, search_type, request_id, existing_tags, new_tags_set)
            if len(new_posts) >= target_count:
                break
    
    if len(new_posts) == target_count:
        save_to_database(new_posts)
    else:
        print(f"Ошибка: не удалось собрать 25 новых постов. Найдено только {len(new_posts)} новых постов.")
    
    if new_tags_set:
        save_new_tags_to_file(new_tags_set)
    else:
        print("Новых уникальных тегов не найдено.")

    return new_posts, user_tag_formatted

if __name__ == "__main__":
    main()