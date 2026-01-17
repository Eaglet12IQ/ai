#!/usr/bin/env python3
"""
Скрипт для очистки метаданных из PNG-файлов.
Удаляет/перезаписывает PNG chunks (включая tEXt, zTXt, iTXt и т.д.),
оставляя только необходимые для изображения данные.

Работает рекурсивно по папке.
Создаёт резервные копии только если явно попросили (--backup).

Использование:
    python clean_png_metadata.py "путь/к/папке" [--backup] [--dry-run]

Примеры:
    python clean_png_metadata.py "D:\finish" --backup
    python clean_png_metadata.py . --dry-run
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import shutil
import numpy as np


def clean_metadata_single_file(
    filepath: Path,
    backup: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Очищает метаданные одного PNG-файла.
    Возвращает True, если файл был изменён (или должен был быть при dry-run).
    """
    if not filepath.is_file() or filepath.suffix.lower() != ".png":
        return False

    try:
        with Image.open(filepath) as img:
            # Если нет info — уже чистый файл
            if not img.info:
                print(f"[skip] нет метаданных → {filepath.name}")
                return False

            # Конвертируем в массив байт → теряем всю метаданную
            data = img.convert("RGB")  # или "RGBA", если нужна прозрачность
            img_clean = Image.fromarray(np.array(data))

        if dry_run:
            print(f"[would clean] {filepath.name}  (было {len(img.info)} элементов info)")
            return True

        # Делаем бэкап, если попросили
        if backup:
            backup_path = filepath.with_suffix(filepath.suffix + ".bak")
            shutil.copy2(filepath, backup_path)
            print(f"[backup] {filepath.name} → {backup_path.name}")

        # Перезаписываем файл без метаданных
        img_clean.save(filepath, "PNG", optimize=True)
        print(f"[cleaned] {filepath.name}")

        return True

    except Exception as e:
        print(f"[error] {filepath.name} → {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Очистка метаданных PNG-файлов")
    parser.add_argument("folder", type=str, help="Папка для обработки (рекурсивно)")
    parser.add_argument("--backup", action="store_true", help="Создавать .bak копии перед изменением")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет очищено, не менять файлы")
    args = parser.parse_args()

    root_path = Path(args.folder).resolve()
    if not root_path.is_dir():
        print(f"Ошибка: {root_path} — не является папкой или не существует.")
        sys.exit(1)

    print(f"Очистка метаданных в: {root_path}")
    if args.dry_run:
        print("Режим dry-run — файлы НЕ будут изменены!\n")
    if args.backup:
        print("Будут создаваться .bak копии перед перезаписью\n")

    cleaned_count = 0
    skipped_count = 0
    error_count = 0

    # Рекурсивный обход
    for file_path in root_path.rglob("*.png"):
        success = clean_metadata_single_file(
            file_path,
            backup=args.backup,
            dry_run=args.dry_run
        )
        if success:
            cleaned_count += 1
        elif file_path.is_file():
            skipped_count += 1
        else:
            error_count += 1

    print("\n" + "═" * 60)
    print(f"Обработано файлов:     {cleaned_count + skipped_count + error_count}")
    print(f"  • очищено:           {cleaned_count}")
    print(f"  • пропущено (чистые):{skipped_count}")
    print(f"  • ошибки:            {error_count}")
    print("═" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Укажите хотя бы путь к папке!")
        print("Пример: python clean_png_metadata.py \"D:\\finish\" --backup")
        sys.exit(1)

    main()