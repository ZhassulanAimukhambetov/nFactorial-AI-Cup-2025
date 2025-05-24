from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QSystemTrayIcon, QMenu, QAction, QApplication, QDialog
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt

from gui.settings_dialog import SettingsDialog
from config.settings import load_settings
from agents.activity_monitor import ActivityMonitor
from agents.attention_analyzer import AttentionAnalyzer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TameWork")
        self.setGeometry(100, 100, 700, 500)
        self.setMinimumSize(400, 300)

        self.current_settings = load_settings()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.status_label = QLabel(self.get_initial_status_message())
        self.status_label.setAlignment(Qt.AlignCenter)

        font = QFont("Inter", 16)
        font.setBold(True)
        self.status_label.setFont(font)

        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f0f4f8;
                border: 2px solid #cbd5e1;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                color: #334155;
            }
        """)
        layout.addWidget(self.status_label)

        # Active Window Label теперь будет показывать и заголовок, и исполняемый файл
        self.active_window_label = QLabel("Активное окно: Неизвестно\nПриложение: Неизвестно")
        self.active_window_label.setAlignment(Qt.AlignCenter)
        self.active_window_label.setFont(QFont("Inter", 12))
        self.active_window_label.setStyleSheet("QLabel { color: #64748b; margin-bottom: 10px; }")
        layout.addWidget(self.active_window_label)

        self.settings_button = QPushButton("Настройки")
        self.settings_button.clicked.connect(self.open_settings)  # type: ignore

        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #4f46e5;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #4338ca;
            }
            QPushButton:pressed {
                background-color: #3730a3;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        """)
        layout.addWidget(self.settings_button)

        layout.addStretch(1)

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon.fromTheme("applications-other", QIcon(":/icons/default.png")))
        self.tray_icon.setToolTip("TameWork")

        tray_menu = QMenu()
        show_action = QAction("Показать окно", self)
        show_action.triggered.connect(self.showNormal)  # type: ignore
        tray_menu.addAction(show_action)

        hide_action = QAction("Скрыть окно", self)
        hide_action.triggered.connect(self.hide)  # type: ignore
        tray_menu.addAction(hide_action)

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close_application)  # type: ignore
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        self.tray_icon.activated.connect(self.on_tray_icon_activated)  # type: ignore

        self.activity_monitor = ActivityMonitor()
        self.attention_analyzer = AttentionAnalyzer()

        # --- ИЗМЕНЕНО: Подключаем сигнал с двумя аргументами ---
        self.activity_monitor.active_window_changed.connect(self.update_active_window_display)
        self.activity_monitor.active_window_changed.connect(self.attention_analyzer.add_data_for_analysis)

        self.activity_monitor.status_message.connect(self.update_status)
        self.attention_analyzer.status_message.connect(self.update_status)

        self.attention_analyzer.analysis_result.connect(self.handle_analysis_result)
        self.attention_analyzer.action_command.connect(self.handle_action_command)

        self.activity_monitor.start()
        self.attention_analyzer.start()

    def get_initial_status_message(self):
        mode_text = "Мягкий" if self.current_settings["mode"] == "soft" else "Строгий"
        return f"TameWork\nРежим: {mode_text}"

    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.current_settings = load_settings()
            self.update_status(self.get_initial_status_message())
            self.attention_analyzer.update_settings()
            print("Настройки обновлены после закрытия диалога.")
        else:
            print("Диалог настроек отменен.")

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.showNormal()
                self.activateWindow()

    def closeEvent(self, event):
        self.activity_monitor.stop()
        self.attention_analyzer.stop()
        if self.tray_icon.isVisible():
            self.hide()
            event.ignore()
            self.tray_icon.showMessage(
                "TameWork",
                "Приложение свернуто в системный трей.",
                QSystemTrayIcon.Information,
                2000
            )
        else:
            event.accept()

    def close_application(self):
        self.activity_monitor.stop()
        self.attention_analyzer.stop()
        self.tray_icon.hide()
        QApplication.quit()

    def update_status(self, message: str):
        self.status_label.setText(message)

    # --- ИЗМЕНЕНО: Слот теперь принимает два аргумента ---
    def update_active_window_display(self, window_title: str, executable_name: str):
        display_title = window_title if window_title else "Нет активного окна"
        display_executable = executable_name if executable_name else "Неизвестно"
        self.active_window_label.setText(f"Активное окно: {display_title}\nПриложение: {display_executable}")

    def handle_analysis_result(self, window_title: str, classification: str):
        if classification == "distraction":
            self.status_label.setText(f"ОТВЛЕЧЕНИЕ: {window_title[:40]}...")
            self.status_label.setStyleSheet(
                "QLabel { background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 20px; "
                "margin-bottom: 20px; color: #dc2626; }")
        elif classification == "productive":
            self.status_label.setText(f"Продуктивно: {window_title[:40]}...")
            self.status_label.setStyleSheet(
                "QLabel { background-color: #ecfdf5; border: 2px solid #22c55e; border-radius: 10px; padding: 20px; "
                "margin-bottom: 20px; color: #16a34a; }")
        elif classification == "ignored":
            self.status_label.setText(f"Игнорируется: {window_title[:40]}...")
            self.status_label.setStyleSheet(
                "QLabel { background-color: #e0f2fe; border: 2px solid #38bdf8; border-radius: 10px; padding: 20px; "
                "margin-bottom: 20px; color: #0ea5e9; }")
        # elif classification == "ollama_down":
        #     self.status_label.setText("ОШИБКА: Ollama не запущена!")
        #     self.status_label.setStyleSheet(
        #         "QLabel { background-color: #fffbeb; border: 2px solid #f59e0b; border-radius: 10px; padding: 20px; "
        #         "margin-bottom: 20px; color: #d97706; }")
        elif classification == "error":
            self.status_label.setText("ОШИБКА АНАЛИЗА!")
            self.status_label.setStyleSheet(
                "QLabel { background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 20px; "
                "margin-bottom: 20px; color: #dc2626; }")
        elif classification == "unknown":
            self.status_label.setText(f"Неизвестно: {window_title[:40]}...")
            self.status_label.setStyleSheet(
                "QLabel { background-color: #eff6ff; border: 2px solid #60a5fa; border-radius: 10px; padding: 20px; "
                "margin-bottom: 20px; color: #2563eb; }")
        else:
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #f0f4f8;
                    border: 2px solid #cbd5e1;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    color: #334155;
                }
            """)

    def handle_action_command(self, action_type: str, target_info: str):
        if action_type == "notify":
            self.tray_icon.showMessage(
                "TameWork: Отвлечение!",
                target_info,
                QSystemTrayIcon.Warning,
                5000
            )
            print(f"Команда: Уведомление. Сообщение: {target_info}")
        elif action_type == "block_or_close":
            print(f"Команда: Блокировать/Закрыть. Цель: {target_info}")
            self.tray_icon.showMessage(
                "TameWork: Блокировка!",
                f"Пытаюсь заблокировать/закрыть: {target_info}",
                QSystemTrayIcon.Critical,
                5000
            )
