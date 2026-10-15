"""
run_snake.py — Entry point launcher for Leeth Snake PyQt6 application.

Usage:
    python run_snake.py
"""

import sys
from PyQt6.QtWidgets import QApplication
from snake_ui_qt import SnakeMainWindow


def main():
    app = QApplication(sys.argv)
    window = SnakeMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
