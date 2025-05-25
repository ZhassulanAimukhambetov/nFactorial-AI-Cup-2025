from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QSystemTrayIcon,
    QMenu, QAction, QApplication, QDialog, QStyle
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

from agents.decision_manager import DecisionManager, Decision
from gui.settings_dialog import SettingsDialog
from config.settings import load_settings
from agents.activity_monitor import ActivityMonitor
from agents.attention_analyzer import AttentionAnalyzer


class StatusWidget(QLabel):
    """Виджет для отображения статуса с предустановленными стилями"""

    STYLES = {
        'default': """
            QLabel {
                background-color: #f0f4f8;
                border: 2px solid #cbd5e1;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                color: #334155;
            }
        """,
        'distraction': """
            QLabel { 
                background-color: #fef2f2; 
                border: 2px solid #ef4444; 
                border-radius: 10px; 
                padding: 20px;
                margin-bottom: 20px; 
                color: #dc2626; 
            }
        """,
        'productive': """
            QLabel { 
                background-color: #ecfdf5; 
                border: 2px solid #22c55e; 
                border-radius: 10px; 
                padding: 20px;
                margin-bottom: 20px; 
                color: #16a34a; 
            }
        """,
        'ignored': """
            QLabel { 
                background-color: #e0f2fe; 
                border: 2px solid #38bdf8; 
                border-radius: 10px; 
                padding: 20px;
                margin-bottom: 20px; 
                color: #0ea5e9; 
            }
        """,
        'error': """
            QLabel { 
                background-color: #fef2f2; 
                border: 2px solid #ef4444; 
                border-radius: 10px; 
                padding: 20px;
                margin-bottom: 20px; 
                color: #dc2626; 
            }
        """,
        'unknown': """
            QLabel { 
                background-color: #eff6ff; 
                border: 2px solid #60a5fa; 
                border-radius: 10px; 
                padding: 20px;
                margin-bottom: 20px; 
                color: #2563eb; 
            }
        """
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        font = QFont("Inter", 16)
        font.setBold(True)
        self.setFont(font)
        self.set_style('default')

    def set_style(self, style_name: str):
        """Применяет предустановленный стиль"""
        if style_name in self.STYLES:
            self.setStyleSheet(self.STYLES[style_name])

    def update_for_classification(self, classification: str, window_title: str = ""):
        """Обновляет статус на основе классификации"""
        classification_map = {
            "distraction": ("distraction", f"ОТВЛЕЧЕНИЕ: {window_title[:40]}..."),
            "productive": ("productive", f"Продуктивно: {window_title[:40]}..."),
            "ignored": ("ignored", f"Игнорируется: {window_title[:40]}..."),
            "error": ("error", "ОШИБКА АНАЛИЗА!"),
            "unknown": ("unknown", f"Неизвестно: {window_title[:40]}...")
        }

        if classification in classification_map:
            style, text = classification_map[classification]
            self.set_style(style)
            self.setText(text)
        else:
            self.set_style('default')


class TrayManager:
    """Менеджер системного трея"""

    def __init__(self, parent_window):
        self.parent = parent_window
        self.tray_icon = QSystemTrayIcon(parent_window)
        self.setup_tray()

    def setup_tray(self):
        """Настройка системного трея"""
        icon = self.parent.style().standardIcon(QStyle.SP_MessageBoxInformation)
        self.tray_icon.setIcon(icon)
        self.tray_icon.setToolTip("TameWork")

        # Создание меню
        tray_menu = QMenu()

        actions = [
            ("Показать окно", self.parent.showNormal),
            ("Скрыть окно", self.parent.hide),
            ("Выход", self.parent.close_application)
        ]

        for text, handler in actions:
            action = QAction(text, self.parent)
            action.triggered.connect(handler)  # type: ignore
            tray_menu.addAction(action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)  # type: ignore
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        """Обработка активации трея"""
        if reason == QSystemTrayIcon.Trigger:
            if self.parent.isVisible():
                self.parent.hide()
            else:
                self.parent.showNormal()
                self.parent.activateWindow()

    def show_message(self, title: str, message: str, icon=QSystemTrayIcon.Information, duration=5000):
        """Показ уведомления в трее"""
        self.tray_icon.showMessage(title, message, icon, duration)


class ActionHandler:
    """Обработчик действий DecisionManager"""

    def __init__(self, parent_window):
        self.parent = parent_window

    def handle_decision(self, decision: Decision):
        """Обрабатывает решение DecisionManager"""
        try:
            handlers = {
                'notify': self._handle_notification,
                'block_app': self._handle_block_app,
                'close_window': self._handle_close_window,
                'log_activity': self._handle_log_activity,
                'no_action': self._handle_no_action
            }

            handler = handlers.get(decision.action_type.value)
            if handler:
                handler(decision)
            else:
                self.parent.update_status(f"Неизвестное действие: {decision.action_type.value}")

        except Exception as e:
            self.parent.update_status(f"Ошибка выполнения действия: {str(e)}")

    def _handle_notification(self, decision: Decision):
        """Обработка уведомлений"""
        self.parent.tray_manager.show_message(
            "TameWork - Уведомление",
            decision.message,
            QSystemTrayIcon.Warning
        )
        self.parent.update_status(f"Уведомление: {decision.message[:50]}...")

    def _handle_block_app(self, decision: Decision):
        """Обработка блокировки приложения"""
        executable = decision.target.split('|')[0].strip() if decision.target else "Неизвестно"
        self.parent.update_status(f"Блокировка: {executable[:20]}...")
        self.parent.tray_manager.show_message(
            "TameWork - Блокировка",
            f"Блокировка приложения: {executable}",
            QSystemTrayIcon.Critical
        )
        # Здесь должна быть реальная логика блокировки

    def _handle_close_window(self, decision: Decision):
        """Обработка закрытия окна"""
        self.parent.update_status("Закрытие активного окна...")
        try:
            import pyautogui
            pyautogui.hotkey('alt', 'f4')
            self.parent.tray_manager.show_message(
                "TameWork - Действие",
                "Активное окно закрыто",
                QSystemTrayIcon.Information
            )
        except Exception as e:
            self.parent.update_status(f"Ошибка закрытия окна: {str(e)}")

    def _handle_log_activity(self, decision: Decision):
        """Обработка логирования активности"""
        self.parent.update_status(decision.reason or "Активность зарегистрирована")

    def _handle_no_action(self, decision: Decision):
        """Обработка отсутствия действий"""
        if decision.reason:
            self.parent.update_status(decision.reason)


class StatisticsManager:
    """Менеджер статистики"""

    def __init__(self, decision_manager, display_widget):
        self.decision_manager = decision_manager
        self.display_widget = display_widget
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_statistics)  # type: ignore
        self.timer.start(5000)  # Обновляем каждые 5 секунд

    def update_statistics(self):
        """Обновление отображения статистики"""
        try:
            stats = self.decision_manager.get_statistics()
            tooltip_text = self._format_statistics(stats)
            self.display_widget.setToolTip(tooltip_text)
        except Exception as e:
            self.display_widget.setToolTip(f"Ошибка получения статистики: {str(e)}")

    def _format_statistics(self, stats: dict) -> str:
        """Форматирование статистики для отображения"""
        text_parts = [
            "📊 Статистика системы:",
            f"Всего решений: {stats['total_decisions']}",
            f"Уведомлений: {stats['performance_stats']['notifications_sent']}",
            f"Блокировок: {stats['performance_stats']['apps_blocked']}",
            "Топ нарушений:"
        ]

        for violation in stats.get('top_violations', [])[:2]:
            text_parts.append(f"- {violation['target']}: {violation['count']} раз")

        text_parts.append(f"\nТренд: {stats.get('decision_trend', 'Нет данных')}")

        return "\n".join(text_parts)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_settings = load_settings()
        self._setup_window()
        self._setup_components()
        self._setup_connections()
        self._start_monitoring()

    def _setup_window(self):
        """Настройка основного окна"""
        self.setWindowTitle("TameWork")
        self.setGeometry(100, 100, 700, 500)
        self.setMinimumSize(400, 300)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Статус виджет
        self.status_widget = StatusWidget()
        self.status_widget.setText(self._get_initial_status_message())
        layout.addWidget(self.status_widget)

        # Информация об активном окне
        self.active_window_label = QLabel("Активное окно: Неизвестно\nПриложение: Неизвестно")
        self.active_window_label.setAlignment(Qt.AlignCenter)
        self.active_window_label.setFont(QFont("Inter", 12))
        self.active_window_label.setStyleSheet("QLabel { color: #64748b; margin-bottom: 10px; }")
        layout.addWidget(self.active_window_label)

        # Кнопка настроек
        self.settings_button = QPushButton("Настройки")
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #4f46e5;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4338ca;
            }
            QPushButton:pressed {
                background-color: #3730a3;
            }
        """)
        layout.addWidget(self.settings_button)
        layout.addStretch(1)

    def _setup_components(self):
        """Инициализация компонентов"""
        # Системный трей
        self.tray_manager = TrayManager(self)

        # Агенты мониторинга
        self.activity_monitor = ActivityMonitor()
        self.attention_analyzer = AttentionAnalyzer()
        self.decision_manager = DecisionManager(self.current_settings)

        # Обработчики
        self.action_handler = ActionHandler(self)
        self.statistics_manager = StatisticsManager(self.decision_manager, self.active_window_label)

    def _setup_connections(self):
        """Настройка соединений сигналов и слотов"""
        # Кнопка настроек
        self.settings_button.clicked.connect(self.open_settings)  # type: ignore

        # ActivityMonitor
        self.activity_monitor.active_window_changed.connect(self.update_active_window_display)
        self.activity_monitor.active_window_changed.connect(self.attention_analyzer.add_data_for_analysis)
        self.activity_monitor.status_message.connect(self.update_status)

        # AttentionAnalyzer
        self.attention_analyzer.analysis_result.connect(self.handle_analysis_result)
        self.attention_analyzer.analysis_result.connect(self._create_decision_from_analysis)
        self.attention_analyzer.screenshot_analysis_result.connect(self.decision_manager.make_screenshot_decision)
        self.attention_analyzer.status_message.connect(self.update_status)

        # DecisionManager
        self.decision_manager.notification_request.connect(self._show_notification)
        self.decision_manager.action_required.connect(self._handle_legacy_action)
        self.decision_manager.status_update.connect(self.update_status)

    def _start_monitoring(self):
        """Запуск мониторинга"""
        self.activity_monitor.start()
        self.attention_analyzer.start()

    def _get_initial_status_message(self) -> str:
        """Получение начального сообщения статуса"""
        mode_text = "Мягкий" if self.current_settings["mode"] == "soft" else "Строгий"
        return f"TameWork\nРежим: {mode_text}"

    @pyqtSlot(str, str)
    def update_active_window_display(self, window_title: str, executable_name: str):
        """Обновление отображения активного окна"""
        display_title = window_title if window_title else "Нет активного окна"
        display_executable = executable_name if executable_name else "Неизвестно"
        self.active_window_label.setText(f"Активное окно: {display_title}\nПриложение: {display_executable}")

    @pyqtSlot(str, str)
    def handle_analysis_result(self, window_title: str, classification: str):
        """Обработка результатов анализа"""
        self.status_widget.update_for_classification(classification, window_title)

    @pyqtSlot(str, str)
    def _create_decision_from_analysis(self, window_title: str, classification: str):
        """Создание решения на основе анализа"""
        analysis_data = {
            'classification': classification,
            'window_title': window_title,
            'executable_name': '',  # Можно получить из activity_monitor
            'confidence': 1.0
        }
        decision = self.decision_manager.make_decision(analysis_data)
        self.action_handler.handle_decision(decision)

    @pyqtSlot(str)
    def _show_notification(self, message: str):
        """Показ уведомлений из DecisionManager"""
        print(f"🔔 УВЕДОМЛЕНИЕ: {message}")
        self.tray_icon.showMessage(
            "TameWork - Уведомление",
            message,
            QSystemTrayIcon.Information,
            5000
        )
        self.update_status(f"Уведомление: {message[:50]}...")
    @pyqtSlot(str, str)
    def _handle_legacy_action(self, action_type: str, target_info: str):
        """Обработка устаревших команд действий (для совместимости)"""
        # Создаем временный объект Decision для совместимости
        from agents.decision_manager import ActionType, ViolationSeverity

        action_map = {
            'block_or_close': ActionType.BLOCK_APP,
            'close_active_window': ActionType.CLOSE_WINDOW,
            'notify': ActionType.NOTIFY
        }

        action = action_map.get(action_type, ActionType.NO_ACTION)
        decision = Decision(
            action_type=action,
            severity=ViolationSeverity.MEDIUM,
            target=target_info,
            message=f"Действие: {action_type}",
            reason="Обработка устаревшей команды"
        )
        self.action_handler.handle_decision(decision)

    def update_status(self, message: str):
        """Обновление статуса"""
        # Если сообщение не содержит специальной классификации, используем обычное обновление
        if not any(keyword in message.lower() for keyword in ['отвлечение', 'продуктивно', 'ошибка']):
            self.status_widget.setText(message)

    def open_settings(self):
        """Открытие диалога настроек"""
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.current_settings = load_settings()
            self.status_widget.setText(self._get_initial_status_message())
            self.attention_analyzer.update_settings()
            self.decision_manager.update_settings(self.current_settings)
            print("Настройки обновлены после закрытия диалога.")
        else:
            print("Диалог настроек отменен.")

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        self.activity_monitor.stop()
        self.attention_analyzer.stop()

        if self.tray_manager.tray_icon.isVisible():
            self.hide()
            event.ignore()
            self._show_notification("Приложение свернуто в системный трай")
        else:
            event.accept()

    def close_application(self):
        """Полное закрытие приложения"""
        self.activity_monitor.stop()
        self.attention_analyzer.stop()
        self.tray_manager.tray_icon.hide()
        QApplication.quit()