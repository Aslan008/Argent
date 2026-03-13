from ui import console
from rich.table import Table
import os

from ui import console
from rich.table import Table
import os
import difflib

def command_todo():
    # Получаем текущую директорию проекта
    project_dir = os.getcwd()

    # Создаем таблицу
    table = Table(title="TODO Items")
    table.add_column("Файл", style="cyan")
    table.add_column("Строка", style="magenta")
    table.add_column("Текст задачи", style="green")

    # Расширение файлов, которые мы хотим сканировать
    code_file_extensions = {".py", ".cs", ".java", ".js", ".ts", ".cpp", ".c", ".h", ".go", ".rb", ".swift", ".kt", ".rs", ".scala"}

    # Проходим по всем файлам в директории проекта
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1]
            if file_extension in code_file_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if line.startswith('TODO: '):
                                task_text = line.strip().replace('TODO: ', '')
                                table.add_row(file_path, str(i + 1), task_text)
                except Exception as e:
                    console.print(f"Ошибка при чтении файла {file_path}: {e}", style="red")

    # Выводим таблицу
    console.print(table)
