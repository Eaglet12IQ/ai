import sqlite3

def reset_autoincrement(table_name, db_path="danbooru_api/danbooru_posts.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Находим максимальный id в таблице
    c.execute(f"SELECT MAX(request_id) FROM {table_name}")
    max_id = c.fetchone()[0]
    if max_id is None:
        max_id = 0
    
    # Обновляем seq в служебной таблице
    c.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = ?", (max_id, table_name))
    conn.commit()
    conn.close()
    print(f"АВТОИНКРЕМЕНТ для {table_name} обновлён. Следующее значение будет {max_id+1}")

import sqlite3

def delete_requests_and_posts(db_path="danbooru_api/danbooru_posts.db", ids_to_delete=(57, 58)):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    try:
        # Удаляем посты с этими request_id
        c.execute(f"DELETE FROM posts WHERE request_id IN ({','.join('?'*len(ids_to_delete))})", ids_to_delete)

        # Удаляем записи из requests
        c.execute(f"DELETE FROM requests WHERE request_id IN ({','.join('?'*len(ids_to_delete))})", ids_to_delete)

        conn.commit()
        print(f"Удалены request_id {ids_to_delete} и связанные с ними посты.")
    except Exception as e:
        print(f"Ошибка при удалении: {e}")
    finally:
        conn.close()

delete_requests_and_posts()
reset_autoincrement("requests")
reset_autoincrement("posts")
