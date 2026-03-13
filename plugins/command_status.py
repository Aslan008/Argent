from ui import console
from datetime import datetime
import os
import platform


def command_status(*args):
    # Получаем текущую дату
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Получаем общее количество файлов в проекте
    project_dir = 'C:\Users\mshat\.argent'  # Замените на путь к вашему проекту
    file_count = sum(1 for root, dirs, files in os.walk(project_dir) for _ in files)

    # Получаем версию Python
    python_version = platform.python_version()

    # Создаем красивый вывод в рамке Panel
    status_panel = console.Panel(
        f"Дата: {current_date}
Количество файлов: {file_count}
Версия Python: {python_version}",
        title="Статус проекта",
        border_style="blue"
    )

    console.print(status_panel)