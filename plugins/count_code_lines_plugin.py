from ui import console
import os

def on_startup():
    global count_code_lines
    count_code_lines = 0


def count_code_lines_in_directory(directory):
    global count_code_lines
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.js', '.java', '.cpp', '.cs', '.rb', '.go', '.ts', '.tsx', '.php', '.html', '.css', '.md')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        count_code_lines += len(lines)
                except Exception as e:
                    console.print(f"[red]Ошибка при чтении файла {file_path}: {str(e)}[/red]")


def on_tool_call(func_name, args):
    if func_name == 'count_code_lines':
        directory = args.get('directory', '.')
        count_code_lines_in_directory(directory)
        console.print(f"[green]Общее количество строк кода: {count_code_lines}[/green]")
        return False
    return True