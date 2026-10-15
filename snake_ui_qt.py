"""
snake_ui_qt.py — Polished visual interface for Snake game played by the Leeth model.

Features:
1. High-FPS custom QPainter canvas with glowing graphics and sleek cyberpunk dark theme.
2. Real-time "Model Thoughts" side-panel showing 4 direction probabilities and comparison with BFS expert.
3. Duel / Comparison Mode: simultaneous or round-by-round comparison between Leeth Model, BFS Expert, and Human player.
4. Step-by-step playback / pause / single-step inspection.
5. Grid modes: 6x6 (In-Domain), 8x8, and 12x12 (Zero-Shot!).
6. Session metrics & leaderboard tracking (win-rate, optimality rate, mean/median length).
7. Human controls (WASD and Arrow keys).
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from leeth_player import LeethSnakePlayer
from snake_engine import SnakeGame, SessionTracker, StepRecord
from src.leeth.task_domains import DIR_ORDER, DIRS, snake_expert_scores


# ---------------------------------------------------------------------------
# Colors and Styling Tokens
# ---------------------------------------------------------------------------
DARK_BG = "#0b0f19"
PANEL_BG = "#131b2e"
CARD_BG = "#1a233a"
BORDER_COLOR = "#2a3756"
TEXT_PRIMARY = "#f8fafc"
TEXT_MUTED = "#94a3b8"

ACCENT_EMERALD = "#10b981"  # Snake / Win / Optimal
ACCENT_CORAL = "#f43f5e"    # Food / Collision / Crash
ACCENT_CYAN = "#06b6d4"     # Model focus / BFS
ACCENT_PURPLE = "#a855f7"   # Secondary accent
ACCENT_AMBER = "#f59e0b"    # Warning / Divergence

GLOBAL_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'SF Pro Display', Roboto, sans-serif;
    font-size: 13px;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    background: {PANEL_BG};
    border-radius: 8px;
}}

QTabBar::tab {{
    background: {CARD_BG};
    color: {TEXT_MUTED};
    padding: 8px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    background: {PANEL_BG};
    color: {ACCENT_CYAN};
    border-bottom: 2px solid {ACCENT_CYAN};
}}

QGroupBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    margin-top: 18px;
    padding-top: 14px;
    font-weight: bold;
    color: {ACCENT_CYAN};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}}

QPushButton {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

QPushButton:hover {{
    background-color: #24304f;
    border-color: {ACCENT_CYAN};
}}

QPushButton:pressed {{
    background-color: #1a2238;
}}

QPushButton#primaryBtn {{
    background-color: {ACCENT_CYAN};
    color: #0b0f19;
    border: none;
}}

QPushButton#primaryBtn:hover {{
    background-color: #22d3ee;
}}

QPushButton#stopBtn {{
    background-color: {ACCENT_CORAL};
    color: #ffffff;
    border: none;
}}

QPushButton#stopBtn:hover {{
    background-color: #fb7185;
}}

QComboBox {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 6px 12px;
    color: {TEXT_PRIMARY};
}}

QComboBox::drop-down {{
    border: none;
}}

QTableWidget {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    gridline-color: {BORDER_COLOR};
    color: {TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER_COLOR};
    padding: 6px;
    font-weight: bold;
    color: {ACCENT_CYAN};
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {BORDER_COLOR};
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT_CYAN};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #ffffff;
    border: 2px solid {ACCENT_CYAN};
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}}
"""


# ---------------------------------------------------------------------------
# Canvas Widget: Smooth Snake Renderer
# ---------------------------------------------------------------------------
class SnakeCanvas(QWidget):
    """Interactive graphic canvas rendering snake field, head, body, and food."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.game: Optional[SnakeGame] = None
        self.setMinimumSize(320, 320)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.last_move_dir = "вправо"

    def set_game(self, game: SnakeGame, last_dir: str = "вправо") -> None:
        self.game = game
        self.last_move_dir = last_dir
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Clear background
        painter.fillRect(self.rect(), QColor(DARK_BG))

        if not self.game:
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Нажмите «Новая игра»")
            return

        w = self.width()
        h = self.height()
        grid_h = self.game.height
        grid_w = self.game.width

        # Keep square aspect ratio
        margin = 16
        avail_w = w - 2 * margin
        avail_h = h - 2 * margin
        cell_size = min(avail_w / grid_w, avail_h / grid_h)

        board_w = cell_size * grid_w
        board_h = cell_size * grid_h
        start_x = margin + (avail_w - board_w) / 2.0
        start_y = margin + (avail_h - board_h) / 2.0

        # Draw board background
        board_rect = QRectF(start_x, start_y, board_w, board_h)
        painter.setPen(QPen(QColor(BORDER_COLOR), 1.5))
        painter.setBrush(QBrush(QColor(PANEL_BG)))
        painter.drawRoundedRect(board_rect, 10, 10)

        # Draw grid cells
        cell_pen = QPen(QColor("#1e293b"), 0.8)
        painter.setPen(cell_pen)
        for r in range(grid_h):
            for c in range(grid_w):
                cx = start_x + c * cell_size
                cy = start_y + r * cell_size
                painter.drawRoundedRect(QRectF(cx + 1, cy + 1, cell_size - 2, cell_size - 2), 4, 4)

        # Draw Food (glowing apple)
        fr, fc = self.game.food
        if 0 <= fr < grid_h and 0 <= fc < grid_w:
            fx = start_x + fc * cell_size + cell_size / 2.0
            fy = start_y + fr * cell_size + cell_size / 2.0
            food_rad = cell_size * 0.38

            # Glow
            glow_grad = QRadialGradient(fx, fy, food_rad * 1.5)
            glow_grad.setColorAt(0.0, QColor(244, 63, 94, 200))
            glow_grad.setColorAt(0.7, QColor(244, 63, 94, 60))
            glow_grad.setColorAt(1.0, QColor(244, 63, 94, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow_grad))
            painter.drawEllipse(QPointF(fx, fy), food_rad * 1.5, food_rad * 1.5)

            # Apple core
            apple_grad = QRadialGradient(fx - food_rad * 0.2, fy - food_rad * 0.2, food_rad)
            apple_grad.setColorAt(0.0, QColor("#fb7185"))
            apple_grad.setColorAt(0.8, QColor(ACCENT_CORAL))
            apple_grad.setColorAt(1.0, QColor("#be123c"))
            painter.setBrush(QBrush(apple_grad))
            painter.drawEllipse(QPointF(fx, fy), food_rad, food_rad)

        # Draw Snake Body
        body = self.game.body
        body_len = len(body)
        for idx in range(body_len - 1, -1, -1):
            br, bc = body[idx]
            bx = start_x + bc * cell_size
            by = start_y + br * cell_size
            t = idx / max(1, body_len - 1)  # 0 at head, 1 at tail

            if idx == 0:
                # Snake Head
                head_grad = QRadialGradient(bx + cell_size / 2, by + cell_size / 2, cell_size * 0.6)
                head_grad.setColorAt(0.0, QColor("#34d399"))
                head_grad.setColorAt(1.0, QColor(ACCENT_EMERALD))
                painter.setBrush(QBrush(head_grad))
                painter.setPen(QPen(QColor("#059669"), 1.5))
                pad = cell_size * 0.08
                head_rect = QRectF(bx + pad, by + pad, cell_size - 2 * pad, cell_size - 2 * pad)
                painter.drawRoundedRect(head_rect, cell_size * 0.35, cell_size * 0.35)

                # Head Eyes
                eye_rad = cell_size * 0.10
                pupil_rad = cell_size * 0.05
                cx_h = bx + cell_size / 2.0
                cy_h = by + cell_size / 2.0
                offset = cell_size * 0.22

                if self.last_move_dir == "вверх":
                    e1 = QPointF(cx_h - offset, cy_h - offset * 0.7)
                    e2 = QPointF(cx_h + offset, cy_h - offset * 0.7)
                elif self.last_move_dir == "вниз":
                    e1 = QPointF(cx_h - offset, cy_h + offset * 0.7)
                    e2 = QPointF(cx_h + offset, cy_h + offset * 0.7)
                elif self.last_move_dir == "влево":
                    e1 = QPointF(cx_h - offset * 0.7, cy_h - offset)
                    e2 = QPointF(cx_h - offset * 0.7, cy_h + offset)
                else:  # вправо
                    e1 = QPointF(cx_h + offset * 0.7, cy_h - offset)
                    e2 = QPointF(cx_h + offset * 0.7, cy_h + offset)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.drawEllipse(e1, eye_rad, eye_rad)
                painter.drawEllipse(e2, eye_rad, eye_rad)
                painter.setBrush(QBrush(QColor("#0f172a")))
                painter.drawEllipse(e1, pupil_rad, pupil_rad)
                painter.drawEllipse(e2, pupil_rad, pupil_rad)
            else:
                # Body segment with emerald gradient
                r_val = int(16 * (1 - t) + 6 * t)
                g_val = int(185 * (1 - t) + 95 * t)
                b_val = int(129 * (1 - t) + 70 * t)
                seg_color = QColor(r_val, g_val, b_val)
                painter.setBrush(QBrush(seg_color))
                painter.setPen(QPen(QColor(r_val // 2, g_val // 2, b_val // 2), 1.0))
                pad = cell_size * (0.12 + 0.08 * t)
                seg_rect = QRectF(bx + pad, by + pad, cell_size - 2 * pad, cell_size - 2 * pad)
                painter.drawRoundedRect(seg_rect, cell_size * 0.3, cell_size * 0.3)

        # Game Over / Win overlay
        if self.game.game_over:
            overlay_col = QColor(11, 15, 25, 210)
            painter.fillRect(board_rect, overlay_col)

            font = QFont("Segoe UI", 16, QFont.Weight.Bold)
            painter.setFont(font)
            if self.game.won:
                painter.setPen(QColor(ACCENT_EMERALD))
                total_cells = self.game.height * self.game.width
                if len(self.game.body) >= total_cells:
                    text = f"👑 АБСОЛЮТНЫЙ РЕКОРД! ПОЛЕ ЗАПОЛНЕНО НА 100% ({total_cells}/{total_cells})!"
                else:
                    text = f"ПОБЕДА! ДЛИНА {len(self.game.body)} (ЦЕЛЬ: {self.game.target_length})"
            else:
                painter.setPen(QColor(ACCENT_CORAL))
                reason = "СТОЛКНОВЕНИЕ СО СТЕНОЙ" if self.game.death_reason == "wall_collision" else "САМОПЕРЕСЕЧЕНИЕ"
                text = f"ИГРА ОКОНЧЕНА ({reason})"

            painter.drawText(board_rect, Qt.AlignmentFlag.AlignCenter, text)


# ---------------------------------------------------------------------------
# Background Inference Thread
# ---------------------------------------------------------------------------
class ModelInferenceWorker(QThread):
    """Asynchronously runs model prediction to prevent GUI freezes."""

    result_ready = pyqtSignal(dict)

    def __init__(self, player: LeethSnakePlayer):
        super().__init__()
        self.player = player
        self.state_text: str = ""

    def request_predict(self, state_text: str) -> None:
        self.state_text = state_text
        if not self.isRunning():
            self.start()

    def run(self) -> None:
        try:
            res = self.player.predict(self.state_text, options=DIR_ORDER, shuffle=False)
            self.result_ready.emit(res)
        except Exception as e:
            # Fallback if something errors
            self.result_ready.emit({"best_move": "вверх", "probabilities": {d: 0.25 for d in DIR_ORDER}, "error": str(e)})


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------
class SnakeMainWindow(QMainWindow):
    """Main window with Single Player mode, Duel mode, and Session Stats."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leeth Snake Lab — Нейросетевая игра mmBERT + DynamicOptionHead")
        self.resize(1120, 780)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # Model & Engine
        self.player: Optional[LeethSnakePlayer] = None
        self.inference_worker: Optional[ModelInferenceWorker] = None
        self.current_game = SnakeGame(height=6, width=6, target_length=12)
        self.tracker = SessionTracker()
        self.last_move = "вправо"
        self.is_paused = True
        self.active_mode = "model"  # "model", "expert", "human"

        # Timer for auto-play
        self.step_timer = QTimer(self)
        self.step_timer.timeout.connect(self._on_timer_tick)
        self.tick_interval_ms = 300

        # Build UI layout
        self._init_ui()

        # Load model in background on startup
        QTimer.singleShot(100, self._load_model_engine)

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        # Top Header Bar
        header = self._build_header_bar()
        main_layout.addWidget(header)

        # Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # Tab 1: Single Player & Model Inspector
        tab_single = self._build_single_player_tab()
        self.tabs.addTab(tab_single, "🎮 Одиночная игра / Инспектор ходов")

        # Tab 2: Duel Mode (Model vs Expert vs Human)
        tab_duel = self._build_duel_tab()
        self.tabs.addTab(tab_duel, "⚔️ Дуэль и Сравнение (3 Игрока)")

        # Tab 3: Session Stats & Benchmarks
        tab_stats = self._build_stats_tab()
        self.tabs.addTab(tab_stats, "📊 Статистика Сессии")

        # Status Bar
        self.status_label = QLabel("Загрузка модели Leeth mmBERT...")
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        main_layout.addWidget(self.status_label)

    def _build_header_bar(self) -> QWidget:
        header = QFrame()
        header.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 8px;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 10, 16, 10)

        title = QLabel("🐍 LEETH SNAKE LAB")
        title.setStyleSheet(f"font-size: 18px; font-weight: 800; color: {ACCENT_EMERALD}; letter-spacing: 1px;")
        hl.addWidget(title)

        subtitle = QLabel("mmBERT + DynamicOptionHead | Zero-Shot Transfer")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; margin-left: 8px;")
        hl.addWidget(subtitle)

        hl.addStretch()

        # Fast status pill
        self.device_pill = QLabel("Device: CPU")
        self.device_pill.setStyleSheet(f"background: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 4px; padding: 4px 8px; color: {ACCENT_CYAN}; font-weight: bold;")
        hl.addWidget(self.device_pill)
        return header

    def _build_single_player_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        # Left Column: Canvas and Play Controls
        left_box = QVBoxLayout()

        self.canvas = SnakeCanvas()
        self.canvas.set_game(self.current_game)
        left_box.addWidget(self.canvas, stretch=1)

        # Controls Toolbar
        controls_group = QGroupBox("Панель управления игрой")
        ctrl_layout = QGridLayout(controls_group)
        ctrl_layout.setContentsMargins(12, 16, 12, 12)
        ctrl_layout.setSpacing(10)

        # Row 0: Mode, Field Size, and Victory Condition
        ctrl_layout.addWidget(QLabel("Управление:"), 0, 0)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["🧠 Модель Leeth (Auto)", "⚡ BFS-Эксперт", "👤 Человек (Стрелки / WASD)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        ctrl_layout.addWidget(self.combo_mode, 0, 1)

        ctrl_layout.addWidget(QLabel("Размер поля:"), 0, 2)
        self.combo_size = QComboBox()
        self.combo_size.addItems(["6×6 (36 клеток)", "8×8 (64 клетки)", "12×12 (144 клетки)"])
        self.combo_size.currentIndexChanged.connect(self._on_size_or_target_changed)
        ctrl_layout.addWidget(self.combo_size, 0, 3)

        ctrl_layout.addWidget(QLabel("Режим игры:"), 1, 0)
        self.combo_target = QComboBox()
        self.combo_target.addItems([
            "🎯 Стандартная цель (12 / 16 / 24)",
            "🏆 До конца! (Заполнить 100% поля)"
        ])
        self.combo_target.currentIndexChanged.connect(self._on_size_or_target_changed)
        ctrl_layout.addWidget(self.combo_target, 1, 1, 1, 3)

        # Row 2: Speed Slider
        ctrl_layout.addWidget(QLabel("Скорость шага:"), 2, 0)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 800)
        self.speed_slider.setValue(300)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        ctrl_layout.addWidget(self.speed_slider, 2, 1, 1, 2)

        self.speed_val_label = QLabel("300 мс")
        self.speed_val_label.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold;")
        ctrl_layout.addWidget(self.speed_val_label, 2, 3)

        # Row 3: Buttons
        btn_box = QHBoxLayout()
        self.btn_play = QPushButton("▶ Старт (Пробел)")
        self.btn_play.setObjectName("primaryBtn")
        self.btn_play.clicked.connect(self._toggle_play_pause)
        btn_box.addWidget(self.btn_play)

        self.btn_step = QPushButton("⏭ Шаг (F10 / S)")
        self.btn_step.clicked.connect(self._manual_step)
        btn_box.addWidget(self.btn_step)

        self.btn_reset = QPushButton("🔄 Сброс (R)")
        self.btn_reset.clicked.connect(self._reset_game)
        btn_box.addWidget(self.btn_reset)

        ctrl_layout.addLayout(btn_box, 3, 0, 1, 4)
        left_box.addWidget(controls_group)
        layout.addLayout(left_box, stretch=3)

        # Right Column: Model Thoughts Panel
        right_box = QVBoxLayout()

        thoughts_group = QGroupBox("Мысли модели (Softmax Probabilities)")
        t_layout = QVBoxLayout(thoughts_group)
        t_layout.setContentsMargins(14, 18, 14, 14)
        t_layout.setSpacing(12)

        self.prob_bars: Dict[str, Tuple[QLabel, QProgressBar]] = {}
        dir_labels = {
            "вверх": "▲ Вверх (-1, 0)",
            "вниз": "▼ Вниз (+1, 0)",
            "влево": "◄ Влево (0, -1)",
            "вправо": "► Вправо (0, +1)",
        }

        for d_key, d_title in dir_labels.items():
            row = QVBoxLayout()
            h_row = QHBoxLayout()
            lbl = QLabel(d_title)
            lbl.setStyleSheet("font-weight: 600;")
            val_lbl = QLabel("0.0%")
            val_lbl.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold;")
            h_row.addWidget(lbl)
            h_row.addStretch()
            h_row.addWidget(val_lbl)

            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(12)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {CARD_BG};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 6px;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_CYAN}, stop:1 {ACCENT_EMERALD});
                    border-radius: 5px;
                }}
            """)
            row.addLayout(h_row)
            row.addWidget(bar)
            t_layout.addLayout(row)
            self.prob_bars[d_key] = (val_lbl, bar)

        # Chosen Move Highlight Card
        self.chosen_card = QFrame()
        self.chosen_card.setStyleSheet(f"background: {CARD_BG}; border: 1px solid {ACCENT_EMERALD}; border-radius: 6px; padding: 8px;")
        c_layout = QHBoxLayout(self.chosen_card)
        c_layout.addWidget(QLabel("ВЫБОР МОДЕЛИ:"))
        self.chosen_text = QLabel("—")
        self.chosen_text.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {ACCENT_EMERALD};")
        c_layout.addWidget(self.chosen_text)
        c_layout.addStretch()
        t_layout.addWidget(self.chosen_card)

        # Expert Comparison Section
        expert_group = QGroupBox("Сравнение с BFS-Экспертом")
        e_layout = QVBoxLayout(expert_group)
        e_layout.setContentsMargins(12, 16, 12, 12)
        e_layout.setSpacing(8)

        self.expert_moves_label = QLabel("Оптимальные ходы BFS: [—]")
        self.expert_moves_label.setStyleSheet("font-weight: 600;")
        e_layout.addWidget(self.expert_moves_label)

        self.verdict_badge = QLabel("ОЖИДАНИЕ ХОДА")
        self.verdict_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_badge.setFixedHeight(30)
        self.verdict_badge.setStyleSheet(f"background: {CARD_BG}; border-radius: 4px; font-weight: bold; color: {TEXT_MUTED};")
        e_layout.addWidget(self.verdict_badge)

        hint = QLabel("У BFS может быть несколько оптимальных ходов (множество). Совпадение фиксируется, если выбор модели входит в это множество.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        e_layout.addWidget(hint)

        t_layout.addWidget(expert_group)
        t_layout.addStretch()

        # Score Tracker Mini Card
        score_card = QFrame()
        score_card.setStyleSheet(f"background: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 10px;")
        sc_layout = QGridLayout(score_card)
        sc_layout.addWidget(QLabel("Текущая длина:"), 0, 0)
        self.lbl_cur_len = QLabel("3")
        self.lbl_cur_len.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {ACCENT_EMERALD};")
        sc_layout.addWidget(self.lbl_cur_len, 0, 1)

        sc_layout.addWidget(QLabel("Цель победы:"), 0, 2)
        self.lbl_target_len = QLabel("12")
        self.lbl_target_len.setStyleSheet("font-size: 16px; font-weight: bold;")
        sc_layout.addWidget(self.lbl_target_len, 0, 3)

        sc_layout.addWidget(QLabel("Шаги:"), 1, 0)
        self.lbl_cur_steps = QLabel("0")
        self.lbl_cur_steps.setStyleSheet("font-weight: bold;")
        sc_layout.addWidget(self.lbl_cur_steps, 1, 1)

        # Board occupancy progress
        sc_layout.addWidget(QLabel("Заполнение:"), 2, 0)
        self.lbl_occupancy = QLabel("3 / 36 (8.3%)")
        self.lbl_occupancy.setStyleSheet(f"font-weight: bold; color: {ACCENT_CYAN};")
        sc_layout.addWidget(self.lbl_occupancy, 2, 1, 1, 3)

        self.occupancy_bar = QProgressBar()
        self.occupancy_bar.setRange(0, 1000)
        self.occupancy_bar.setValue(83)
        self.occupancy_bar.setTextVisible(False)
        self.occupancy_bar.setFixedHeight(8)
        self.occupancy_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {PANEL_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_CYAN}, stop:1 {ACCENT_EMERALD});
                border-radius: 3px;
            }}
        """)
        sc_layout.addWidget(self.occupancy_bar, 3, 0, 1, 4)

        t_layout.addWidget(score_card)

        right_box.addWidget(thoughts_group)
        layout.addLayout(right_box, stretch=2)
        return widget

    def _build_duel_tab(self) -> QWidget:
        """Duel Tab: side-by-side or round-by-round comparison between Model, Expert, and Human."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        info = QLabel("РЕЖИМ ДУЭЛИ: Сравнение трёх игроков на одинаковых полях и сидах спавна еды")
        info.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {ACCENT_CYAN};")
        layout.addWidget(info)

        # Leaderboard Table
        self.duel_table = QTableWidget()
        self.duel_table.setColumnCount(6)
        self.duel_table.setHorizontalHeaderLabels(["Игрок", "Игр", "Win-Rate", "Средняя длина", "Медианная длина", "Оптимальность ходов"])
        self.duel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.duel_table.setRowCount(3)

        players = [
            ("🧠 Модель Leeth (mmBERT)", ACCENT_CYAN),
            ("⚡ BFS-Эксперт (Идеальный)", ACCENT_EMERALD),
            ("👤 Человек (WASD / Стрелки)", ACCENT_PURPLE),
        ]
        for row, (p_name, col) in enumerate(players):
            item = QTableWidgetItem(p_name)
            item.setForeground(QColor(col))
            self.duel_table.setItem(row, 0, item)
            for c in range(1, 6):
                self.duel_table.setItem(row, c, QTableWidgetItem("—"))

        layout.addWidget(self.duel_table)

        # Controls for Duel
        d_ctrl = QHBoxLayout()
        self.btn_run_duel = QPushButton("🚀 Запустить тестовую серию Дуэли (10 раундов Модель vs Эксперт)")
        self.btn_run_duel.setObjectName("primaryBtn")
        self.btn_run_duel.clicked.connect(self._run_auto_duel_series)
        d_ctrl.addWidget(self.btn_run_duel)

        self.btn_clear_duel = QPushButton("Сбросить таблицу дуэли")
        self.btn_clear_duel.clicked.connect(self._clear_duel_stats)
        d_ctrl.addWidget(self.btn_clear_duel)
        layout.addLayout(d_ctrl)

        # Duel progress text
        self.duel_log = QLabel("Нажмите «Запустить тестовую серию», чтобы провести дуэль Модель vs BFS на 10 одинаковых сидах.")
        self.duel_log.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(self.duel_log)
        layout.addStretch()
        return widget

    def _build_stats_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        # KPI Cards Row
        cards_layout = QHBoxLayout()
        self.kpi_games = self._create_kpi_card("Сыграно игр", "0")
        self.kpi_winrate = self._create_kpi_card("Win-Rate Модели", "0.0%", target="Цель: ≥85%")
        self.kpi_mean_len = self._create_kpi_card("Средняя длина", "0.0")
        self.kpi_median_len = self._create_kpi_card("Медиана длины", "0.0")
        self.kpi_optimality = self._create_kpi_card("Оптимальность ходов", "0.0%", target="Цель: ~98%")

        for c in [self.kpi_games, self.kpi_winrate, self.kpi_mean_len, self.kpi_median_len, self.kpi_optimality]:
            cards_layout.addWidget(c)
        layout.addLayout(cards_layout)

        # Game History Table
        lbl = QLabel("История сыгранных сессий:")
        lbl.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(lbl)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["№ Игры", "Игрок", "Размер", "Финальная длина", "Шагов", "Результат"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.history_table, stretch=1)
        return widget

    def _create_kpi_card(self, title: str, init_val: str, target: Optional[str] = None) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"background: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 12px;")
        cl = QVBoxLayout(card)
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        cl.addWidget(t_lbl)

        val_lbl = QLabel(init_val)
        val_lbl.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {ACCENT_CYAN};")
        card.value_label = val_lbl  # type: ignore
        cl.addWidget(val_lbl)

        if target:
            tgt_lbl = QLabel(target)
            tgt_lbl.setStyleSheet(f"color: {ACCENT_EMERALD}; font-size: 10px;")
            cl.addWidget(tgt_lbl)
        return card

    # -----------------------------------------------------------------------
    # Model Loading & Worker
    # -----------------------------------------------------------------------
    def _load_model_engine(self) -> None:
        try:
            self.status_label.setText("Загрузка весов mmBERT + DynamicOptionHead (на CPU)...")
            self.player = LeethSnakePlayer(device="cpu")
            self.inference_worker = ModelInferenceWorker(self.player)
            self.inference_worker.result_ready.connect(self._on_model_result)
            self.status_label.setText("Модель готова к инференсу! Нажмите «Старт» для автоигры.")
            self.device_pill.setText(f"Device: {self.player.device.upper()}")
            # Initial prediction for first board state
            self._request_model_thoughts()
        except Exception as e:
            self.status_label.setText(f"Ошибка загрузки модели: {e}")

    # -----------------------------------------------------------------------
    # Game Loop & Handlers
    # -----------------------------------------------------------------------
    def _toggle_play_pause(self) -> None:
        if self.is_paused:
            self.is_paused = False
            self.btn_play.setText("⏸ Пауза (Пробел)")
            self.btn_play.setObjectName("stopBtn")
            self.btn_play.setStyleSheet("")  # re-eval style
            self.step_timer.start(self.tick_interval_ms)
        else:
            self.is_paused = True
            self.btn_play.setText("▶ Старт (Пробел)")
            self.btn_play.setObjectName("primaryBtn")
            self.btn_play.setStyleSheet("")
            self.step_timer.stop()

    def _on_speed_changed(self, val: int) -> None:
        self.tick_interval_ms = val
        self.speed_val_label.setText(f"{val} мс")
        if self.step_timer.isActive():
            self.step_timer.setInterval(val)

    def _on_mode_changed(self, idx: int) -> None:
        modes = ["model", "expert", "human"]
        self.active_mode = modes[idx]
        if self.active_mode == "human":
            # Pause auto timer if human wants to play
            if not self.is_paused:
                self._toggle_play_pause()

    def _on_size_or_target_changed(self) -> None:
        size_idx = self.combo_size.currentIndex()
        target_mode_idx = self.combo_target.currentIndex()

        sizes = [(6, 6), (8, 8), (12, 12)]
        h, w = sizes[size_idx]

        if target_mode_idx == 0:
            default_targets = {(6, 6): 12, (8, 8): 16, (12, 12): 24}
            tgt = default_targets[(h, w)]
        else:
            tgt = h * w  # 100% full board!

        self.current_game = SnakeGame(height=h, width=w, target_length=tgt)
        self.canvas.set_game(self.current_game)
        self.lbl_target_len.setText(str(tgt))
        self._update_scoreboard()
        self._request_model_thoughts()

    def _reset_game(self) -> None:
        self.current_game.reset()
        self.canvas.set_game(self.current_game)
        self._update_scoreboard()
        self._request_model_thoughts()

    def _on_timer_tick(self) -> None:
        if self.current_game.game_over:
            self.tracker.record_game(self.current_game)
            self._append_history_row(self.current_game, self._get_mode_display_name())
            self._update_kpi_cards()
            # Auto-restart next game after brief pause
            self.current_game.reset()
            self.canvas.set_game(self.current_game)
            self._update_scoreboard()

        self._execute_step()

    def _manual_step(self) -> None:
        if not self.current_game.game_over:
            self._execute_step()

    def _execute_step(self) -> None:
        if self.current_game.game_over:
            return

        if self.active_mode == "model":
            # If model is loaded, predict synchronously or use cached thoughts
            if self.player:
                st = self.current_game.get_state_text()
                pred = self.player.predict(st, options=DIR_ORDER, shuffle=False)
                move = pred["best_move"]
                self._update_model_thoughts_ui(pred)
            else:
                move = "вверх"
        elif self.active_mode == "expert":
            opt_moves = self.current_game.get_optimal_moves()
            move = opt_moves[0] if opt_moves else "вверх"
            self._request_model_thoughts()
        else:  # human
            move = self.last_move

        self.last_move = move
        rec = self.current_game.step(move)
        self.canvas.set_game(self.current_game, last_dir=move)
        self._update_scoreboard()

        if self.active_mode != "model":
            self._request_model_thoughts()

    def _request_model_thoughts(self) -> None:
        if self.player and not self.current_game.game_over:
            st = self.current_game.get_state_text()
            if self.inference_worker:
                self.inference_worker.request_predict(st)

    def _on_model_result(self, result: dict) -> None:
        self._update_model_thoughts_ui(result)

    def _update_model_thoughts_ui(self, result: dict) -> None:
        probs = result.get("probabilities", {})
        best = result.get("best_move", "—")
        self.chosen_text.setText(best.upper())

        # Update bars
        for d, p in probs.items():
            if d in self.prob_bars:
                val_lbl, bar = self.prob_bars[d]
                pct = p * 100.0
                val_lbl.setText(f"{pct:.1f}%")
                bar.setValue(int(p * 1000))

        # Expert verdict
        opt_moves = self.current_game.get_optimal_moves()
        self.expert_moves_label.setText(f"Оптимальные ходы BFS: {opt_moves}")

        if best in opt_moves:
            self.verdict_badge.setText("🟢 СОВПАЛ С ЭКСПЕРТОМ (ОПТИМАЛЬНО)")
            self.verdict_badge.setStyleSheet(f"background: #064e3b; color: {ACCENT_EMERALD}; border-radius: 4px; font-weight: bold;")
        else:
            self.verdict_badge.setText("🟡 РАСХОЖДЕНИЕ С ЭКСПЕРТОМ")
            self.verdict_badge.setStyleSheet(f"background: #78350f; color: {ACCENT_AMBER}; border-radius: 4px; font-weight: bold;")

    def _update_scoreboard(self) -> None:
        cur_len = len(self.current_game.body)
        total_cells = self.current_game.height * self.current_game.width
        pct = (cur_len / total_cells) * 100.0

        self.lbl_cur_len.setText(str(cur_len))
        self.lbl_cur_steps.setText(str(self.current_game.steps_count))
        self.lbl_occupancy.setText(f"{cur_len} / {total_cells} ({pct:.1f}%)")
        self.occupancy_bar.setValue(int(pct * 10))

    def _get_mode_display_name(self) -> str:
        if self.active_mode == "model":
            return "🧠 Модель Leeth"
        elif self.active_mode == "expert":
            return "⚡ BFS-Эксперт"
        return "👤 Человек"

    # -----------------------------------------------------------------------
    # Keyboard Event for Human Play & Shortcuts
    # -----------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space:
            self._toggle_play_pause()
            return
        elif key in (Qt.Key.Key_F10, Qt.Key.Key_S):
            self._manual_step()
            return
        elif key == Qt.Key.Key_R:
            self._reset_game()
            return

        # Human direction controls (WASD & Arrows)
        if self.active_mode == "human":
            if key in (Qt.Key.Key_Up, Qt.Key.Key_W):
                self.last_move = "вверх"
                self._manual_step()
            elif key in (Qt.Key.Key_Down, Qt.Key.Key_S):
                self.last_move = "вниз"
                self._manual_step()
            elif key in (Qt.Key.Key_Left, Qt.Key.Key_A):
                self.last_move = "влево"
                self._manual_step()
            elif key in (Qt.Key.Key_Right, Qt.Key.Key_D):
                self.last_move = "вправо"
                self._manual_step()

    # -----------------------------------------------------------------------
    # Stats & History Updates
    # -----------------------------------------------------------------------
    def _update_kpi_cards(self) -> None:
        s = self.tracker.summary()
        self.kpi_games.value_label.setText(str(s["games_played"]))  # type: ignore
        self.kpi_winrate.value_label.setText(f"{s['win_rate']:.1f}%")  # type: ignore
        self.kpi_mean_len.value_label.setText(f"{s['mean_length']:.2f}")  # type: ignore
        self.kpi_median_len.value_label.setText(f"{s['median_length']:.1f}")  # type: ignore
        self.kpi_optimality.value_label.setText(f"{s['optimality_rate']:.1f}%")  # type: ignore

    def _append_history_row(self, game: SnakeGame, player_name: str) -> None:
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        res_str = "ПОБЕДА" if game.won else f"ПОРАЖЕНИЕ ({game.death_reason})"
        res_item = QTableWidgetItem(res_str)
        res_item.setForeground(QColor(ACCENT_EMERALD if game.won else ACCENT_CORAL))

        self.history_table.setItem(row, 0, QTableWidgetItem(f"#{game.steps_count}"))
        self.history_table.setItem(row, 1, QTableWidgetItem(player_name))
        self.history_table.setItem(row, 2, QTableWidgetItem(f"{game.height}x{game.width}"))
        self.history_table.setItem(row, 3, QTableWidgetItem(str(len(game.body))))
        self.history_table.setItem(row, 4, QTableWidgetItem(str(game.steps_count)))
        self.history_table.setItem(row, 5, res_item)
        self.history_table.scrollToBottom()

    # -----------------------------------------------------------------------
    # Duel Series Simulation
    # -----------------------------------------------------------------------
    def _run_auto_duel_series(self) -> None:
        if not self.player:
            self.duel_log.setText("Подождите загрузки модели!")
            return

        self.duel_log.setText("Симуляция 10 одинаковых раундов для Модели и BFS-Эксперта...")
        QApplication.processEvents()

        m_tracker = SessionTracker()
        e_tracker = SessionTracker()

        n_rounds = 10
        for i in range(1, n_rounds + 1):
            seed = 7000 + i
            # Model game
            mg = SnakeGame(height=6, width=6, target_length=12, seed=seed)
            while not mg.game_over and mg.steps_count < 150:
                st = mg.get_state_text()
                p = self.player.predict(st, options=DIR_ORDER, shuffle=False)
                mg.step(p["best_move"])
            m_tracker.record_game(mg)

            # Expert game
            eg = SnakeGame(height=6, width=6, target_length=12, seed=seed)
            while not eg.game_over and eg.steps_count < 150:
                opts = eg.get_optimal_moves()
                eg.step(opts[0] if opts else "вверх")
            e_tracker.record_game(eg)

        # Update duel table
        sm = m_tracker.summary()
        se = e_tracker.summary()

        self._set_duel_row(0, sm)
        self._set_duel_row(1, se)
        self.duel_log.setText(f"Дуэль завершена! Модель win-rate: {sm['win_rate']}%, BFS win-rate: {se['win_rate']}%.")

    def _set_duel_row(self, row: int, summary: dict) -> None:
        self.duel_table.setItem(row, 1, QTableWidgetItem(str(summary["games_played"])))
        self.duel_table.setItem(row, 2, QTableWidgetItem(f"{summary['win_rate']:.1f}%"))
        self.duel_table.setItem(row, 3, QTableWidgetItem(f"{summary['mean_length']:.2f}"))
        self.duel_table.setItem(row, 4, QTableWidgetItem(f"{summary['median_length']:.1f}"))
        self.duel_table.setItem(row, 5, QTableWidgetItem(f"{summary['optimality_rate']:.1f}%"))

    def _clear_duel_stats(self) -> None:
        for r in range(3):
            for c in range(1, 6):
                self.duel_table.setItem(r, c, QTableWidgetItem("—"))
        self.duel_log.setText("Таблица очищена.")


def main():
    app = QApplication(sys.argv)
    window = SnakeMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
