import json
import os

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.attention_booster_settings.json')

DEFAULT_SETTINGS = {
    "distracting_keywords": [
        "youtube.com", "tiktok.com", "twitter.com", "vk.com", "facebook.com",
        "instagram.com", "telegram desktop", "discord", "steam", "netflix", "reddit.com",
        "chrome.exe", "firefox.exe", "msedge.exe", # Браузеры могут быть отвлекающими
        "telegram.exe", "discord.exe" # Исполняемые файлы мессенджеров
    ],
    "productive_keywords": [
        "visual studio code", "pycharm", "jupyter notebook", "terminal", "git bash",
        "powershell", "cmd", "microsoft word", "microsoft excel", "google docs",
        "search", "start",
        "google sheets", "notion", "jira", "github", "gitlab", "stack overflow",
        "nfactorial-ai-cup-2025", # Название вашего проекта
        "code.exe", "pycharm64.exe", "idea64.exe", "clion64.exe", # Исполняемые файлы IDE
        "notepad++.exe", "sublime_text.exe", "codeblocks.exe", # Текстовые редакторы/IDE
        "cmd.exe", "powershell.exe", "bash.exe", # Терминалы
        "python.exe", "javaw.exe", "node.exe" # Процессы выполнения кода
    ],
    "ignored_keywords": [
        "усилитель внимания", # Само приложение
        "settings", # Окно настроек
        "program manager", # Системные окна
        "taskbar",
        "desktop",
        "explorer.exe", # Проводник Windows
        "start menu",
        "windows security",
        "microsoft text input application",
        "searchapp.exe", # Поиск Windows
        "shellexperiencehost.exe", # Оболочка Windows
        "sihost.exe", # Host для Shell Infrastructure Host
        "fontdrvhost.exe", # Хост драйвера шрифтов
        "ctfmon.exe", # Языковая панель
        "dwm.exe", # Диспетчер окон рабочего стола
        "audiodg.exe", # Изоляция графа аудиоустройства
        "systemsettings.exe", # Настройки Windows
        "msiexec.exe", # Установщик Windows
        "runtimebroker.exe", # Вспомогательный процесс Windows
        "dllhost.exe", # Generic Host Process for Win32 Services
        "spoolsv.exe", # Диспетчер печати
        "svchost.exe" # Хост-процесс для служб Windows
    ],
    "mode": "soft",
    "work_start_time": "09:00",
    "work_end_time": "18:00",
    "screenshot_analysis_enabled": True,
    "screenshot_analysis_interval": 15,  # секунды
    "distraction_threshold": 2,
}


def load_settings():
    """
    Загружает настройки из JSON-файла.
    Если файл не существует или поврежден, возвращает дефолтные настройки.
    """
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                return {**DEFAULT_SETTINGS, **settings}
        except json.JSONDecodeError:
            print(f"Ошибка чтения файла настроек: {SETTINGS_FILE}. Используем дефолтные.")
            return DEFAULT_SETTINGS
    else:
        print("Файл настроек не найден. Используем дефолтные настройки.")
        return DEFAULT_SETTINGS


def save_settings(settings: dict):
    """
    Сохраняет текущие настройки в JSON-файл.
    """
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        print("Настройки успешно сохранены.")
    except IOError as e:
        print(f"Ошибка при сохранении настроек: {e}")

