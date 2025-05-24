from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QRadioButton, QButtonGroup, QTimeEdit, QGroupBox, QMessageBox, QTabWidget, QWidget
)
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtGui import QFont

from config.settings import load_settings, save_settings, DEFAULT_SETTINGS


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки Усилителя Внимания")
        self.setGeometry(200, 200, 600, 700)  # Увеличим размер для вкладок
        self.setModal(True)

        self.current_settings = load_settings()

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Используем QTabWidget для организации настроек по вкладкам
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Вкладка "Отвлекающие" ---
        distracting_tab = QWidget()
        distracting_layout = QVBoxLayout(distracting_tab)
        self._add_keyword_section(distracting_layout, "Отвлекающие ресурсы (ключевые слова)",
                                  "distracting_keywords", "keyword_input_distracting", "keyword_list_distracting")
        self.tab_widget.addTab(distracting_tab, "Отвлекающие")

        # --- Вкладка "Продуктивные" ---
        productive_tab = QWidget()
        productive_layout = QVBoxLayout(productive_tab)
        self._add_keyword_section(productive_layout, "Продуктивные ресурсы (ключевые слова)",
                                  "productive_keywords", "keyword_input_productive", "keyword_list_productive")
        self.tab_widget.addTab(productive_tab, "Продуктивные")

        # --- Вкладка "Игнорируемые" ---
        ignored_tab = QWidget()
        ignored_layout = QVBoxLayout(ignored_tab)
        self._add_keyword_section(ignored_layout, "Игнорируемые ресурсы (ключевые слова)",
                                  "ignored_keywords", "keyword_input_ignored", "keyword_list_ignored")
        self.tab_widget.addTab(ignored_tab, "Игнорируемые")

        # --- Вкладка "Общие" ---
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        # Группа "Режим работы"
        mode_group = QGroupBox("Режим работы")
        mode_layout = QVBoxLayout()
        self.mode_button_group = QButtonGroup(self)
        self.soft_mode_radio = QRadioButton("Мягкий (только уведомления)")
        self.mode_button_group.addButton(self.soft_mode_radio)
        mode_layout.addWidget(self.soft_mode_radio)
        self.strict_mode_radio = QRadioButton("Строгий (автоматическая блокировка/закрытие)")
        self.mode_button_group.addButton(self.strict_mode_radio)
        mode_layout.addWidget(self.strict_mode_radio)
        if self.current_settings["mode"] == "soft":
            self.soft_mode_radio.setChecked(True)
        else:
            self.strict_mode_radio.setChecked(True)
        mode_group.setLayout(mode_layout)
        general_layout.addWidget(mode_group)

        # Группа "Рабочие интервалы"
        time_group = QGroupBox("Рабочие интервалы (время блокировки)")
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("С:"))
        self.start_time_edit = QTimeEdit()
        self.start_time_edit.setDisplayFormat("HH:mm")
        start_qtime = QTime.fromString(self.current_settings["work_start_time"], "HH:mm")
        self.start_time_edit.setTime(start_qtime)
        time_layout.addWidget(self.start_time_edit)
        time_layout.addWidget(QLabel("До:"))
        self.end_time_edit = QTimeEdit()
        self.end_time_edit.setDisplayFormat("HH:mm")
        end_qtime = QTime.fromString(self.current_settings["work_end_time"], "HH:mm")
        self.end_time_edit.setTime(end_qtime)
        time_layout.addWidget(self.end_time_edit)
        time_group.setLayout(time_layout)
        general_layout.addWidget(time_group)

        general_layout.addStretch(1)  # Заполнитель для выравнивания
        self.tab_widget.addTab(general_tab, "Общие")

        # --- Кнопки сохранения/отмены ---
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        save_button = QPushButton("Сохранить")
        save_button.clicked.connect(self.save_and_close)
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #16a34a;
            }
        """)
        button_layout.addWidget(save_button)

        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Применяем общий стиль для GroupBox и QLabel
        self.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #1e293b;
                margin-top: 10px;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QLabel {
                font-size: 14px;
                color: #334155;
            }
            QLineEdit, QTimeEdit {
                padding: 8px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-size: 14px;
            }
            QListWidget {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 5px;
                min-height: 100px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #e0e7ff;
                color: #4f46e5;
            }
            QTabWidget::pane { /* The tab widget frame */
                border-top: 1px solid #c2c7cb;
            }
            QTabWidget::tab-bar {
                left: 5px; /* move to the right */
            }
            QTabBar::tab {
                background: #e2e8f0;
                border: 1px solid #c2c7cb;
                border-bottom-color: #c2c7cb; /* same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 8ex;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
                color: #334155;
            }
            QTabBar::tab:selected {
                background: white;
                border-color: #c2c7cb;
                border-bottom-color: white; /* make the bottom line transparent */
            }
            QTabBar::tab:hover {
                background: #cbd5e1;
            }
        """)

        font = QFont("Inter")
        self.setFont(font)

    def _add_keyword_section(self, parent_layout, group_title, settings_key, input_attr_name, list_attr_name):
        """
        Вспомогательный метод для создания секции с ключевыми словами (ввод, список, кнопки).
        """
        group_box = QGroupBox(group_title)
        group_layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        keyword_input = QLineEdit()
        keyword_input.setPlaceholderText("Добавить ключевое слово (например, название окна, URL)")
        setattr(self, input_attr_name, keyword_input)  # Сохраняем ссылку на QLineEdit
        input_layout.addWidget(keyword_input)

        add_button = QPushButton("Добавить")
        # Используем лямбда-функцию для передачи аргументов в слот
        add_button.clicked.connect(lambda: self._add_keyword(settings_key, input_attr_name, list_attr_name))
        input_layout.addWidget(add_button)
        group_layout.addLayout(input_layout)

        keyword_list = QListWidget()
        keyword_list.setSelectionMode(QListWidget.SingleSelection)
        setattr(self, list_attr_name, keyword_list)  # Сохраняем ссылку на QListWidget
        self._refresh_keyword_list(settings_key, list_attr_name)  # Заполняем список
        group_layout.addWidget(keyword_list)

        remove_button = QPushButton("Удалить выбранное")
        remove_button.clicked.connect(lambda: self._remove_keyword(settings_key, list_attr_name))
        group_layout.addWidget(remove_button)

        group_box.setLayout(group_layout)
        parent_layout.addWidget(group_box)

    def _refresh_keyword_list(self, settings_key, list_attr_name):
        """Обновляет QListWidget на основе текущих ключевых слов в настройках."""
        keyword_list_widget = getattr(self, list_attr_name)
        keyword_list_widget.clear()
        for keyword in self.current_settings[settings_key]:
            keyword_list_widget.addItem(keyword)

    def _add_keyword(self, settings_key, input_attr_name, list_attr_name):
        """Добавляет новое ключевое слово в список."""
        keyword_input = getattr(self, input_attr_name)
        new_keyword = keyword_input.text().strip()
        if new_keyword and new_keyword.lower() not in [k.lower() for k in self.current_settings[settings_key]]:
            self.current_settings[settings_key].append(new_keyword)
            self._refresh_keyword_list(settings_key, list_attr_name)
            keyword_input.clear()
        elif new_keyword:
            QMessageBox.warning(self, "Предупреждение", "Это ключевое слово уже есть в списке.")

    def _remove_keyword(self, settings_key, list_attr_name):
        """Удаляет выбранное ключевое слово из списка."""
        keyword_list_widget = getattr(self, list_attr_name)
        selected_items = keyword_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Информация", "Выберите ключевое слово для удаления.")
            return

        reply = QMessageBox.question(self, "Подтверждение",
                                     "Вы уверены, что хотите удалить выбранное ключевое слово?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for item in selected_items:
                self.current_settings[settings_key].remove(item.text())
            self._refresh_keyword_list(settings_key, list_attr_name)

    def save_and_close(self):
        """Сохраняет настройки и закрывает диалог."""
        # Обновляем режим
        if self.soft_mode_radio.isChecked():
            self.current_settings["mode"] = "soft"
        else:
            self.current_settings["mode"] = "strict"

        # Обновляем время
        self.current_settings["work_start_time"] = self.start_time_edit.time().toString("HH:mm")
        self.current_settings["work_end_time"] = self.end_time_edit.time().toString("HH:mm")

        # Сохраняем настройки в файл
        save_settings(self.current_settings)
        self.accept()
