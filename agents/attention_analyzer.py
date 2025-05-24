import requests
import json
import time
from PyQt5.QtCore import QThread, pyqtSignal, QTime

from config.settings import load_settings  # Убедитесь, что load_settings загружает новые ключи


class AttentionAnalyzer(QThread):
    # Сигнал может остаться прежним, если основная информация для пользователя - заголовок
    analysis_result = pyqtSignal(str, str)  # (window_title, classification)
    action_command = pyqtSignal(str, str)
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3.2:3b"  # Убедитесь, что это актуальная и рабочая модель
        self.last_analysis_time = 0
        self.analysis_cooldown = 5
        self.current_settings = load_settings()

        self.window_data_queue = []  # Переименовано для ясности, будет хранить (title, exe_name)
        self.processing_data = False  # Переименовано для ясности

    def update_settings(self):
        self.current_settings = load_settings()
        print("AttentionAnalyzer: Настройки обновлены.")

    def add_data_for_analysis(self, window_title: str, executable_name: str):
        # Нормализуем пустые строки для консистентного сравнения
        current_data = (window_title or "", executable_name or "")

        # Не добавляем, если и заголовок, и имя исполняемого файла пусты
        if not current_data[0] and not current_data[1]:
            return

        if not self.window_data_queue or self.window_data_queue[-1] != current_data:
            self.window_data_queue.append(current_data)
            # print(f"AttentionAnalyzer: Данные добавлены в очередь: {current_data}")

    def run(self):
        self.status_message.emit("Анализатор внимания запущен. Ожидание данных...")

        while self._running:
            if self.window_data_queue and not self.processing_data:
                current_time = time.time()
                if current_time - self.last_analysis_time >= self.analysis_cooldown:
                    data_to_analyze = self.window_data_queue.pop()
                    self.window_data_queue.clear()  # Очищаем, чтобы обрабатывать только самое последнее

                    title_to_analyze, exe_to_analyze = data_to_analyze
                    self.processing_data = True
                    self.status_message.emit(f"Анализ: '{title_to_analyze[:30]}...' ({exe_to_analyze})")

                    classification = self._pre_classify_with_keywords(title_to_analyze, exe_to_analyze)

                    if classification == "needs_llm":
                        classification = self._classify_with_ollama(title_to_analyze, exe_to_analyze)

                    # analysis_result можно расширить, если нужно передавать и exe_name дальше
                    self.analysis_result.emit(title_to_analyze, classification)  # Передаем оригинальный заголовок
                    self._route_action(title_to_analyze, exe_to_analyze,
                                       classification)  # Передаем и exe_name для полноты

                    self.last_analysis_time = time.time()
                    self.processing_data = False
                else:
                    pass
            time.sleep(0.5)

    def _pre_classify_with_keywords(self, window_title: str, executable_name: str) -> str:
        lower_title = window_title.lower()
        # Имя исполняемого файла может быть None или пустым
        lower_exe = executable_name.lower() if executable_name else ""

        # Порядок важен: сначала более специфичные (ignored), затем productive/distracting.
        # Исполняемые файлы часто более надежный индикатор, чем заголовок.

        # --- Игнорируемые ---
        if lower_exe:
            for keyword in self.current_settings.get("ignored_executables", []):
                if keyword.lower() in lower_exe:
                    print(f"AttentionAnalyzer: '{executable_name}' классифицирован как IGNORED по exe.")
                    return "ignored"
        for keyword in self.current_settings.get("ignored_keywords", []):
            if keyword.lower() in lower_title:
                print(f"AttentionAnalyzer: '{window_title}' классифицирован как IGNORED по заголовку.")
                return "ignored"

        # --- Продуктивные ---
        if lower_exe:
            for keyword in self.current_settings.get("productive_executables", []):
                if keyword.lower() in lower_exe:
                    print(f"AttentionAnalyzer: '{executable_name}' классифицирован как PRODUCTIVE по exe.")
                    return "productive"
        # Если exe не продуктивный, проверяем заголовок (например, chrome.exe может быть продуктивным для док-ов)
        for keyword in self.current_settings.get("productive_keywords", []):
            if keyword.lower() in lower_title:
                print(f"AttentionAnalyzer: '{window_title}' классифицирован как PRODUCTIVE по заголовку.")
                return "productive"

        # --- Отвлекающие ---
        if lower_exe:
            for keyword in self.current_settings.get("distracting_executables", []):
                if keyword.lower() in lower_exe:
                    print(f"AttentionAnalyzer: '{executable_name}' классифицирован как DISTRACTION по exe.")
                    return "distraction"
        for keyword in self.current_settings.get("distracting_keywords", []):
            if keyword.lower() in lower_title:
                print(f"AttentionAnalyzer: '{window_title}' классифицирован как DISTRACTION по заголовку.")
                return "distraction"

        return "needs_llm"

    def _classify_with_ollama(self, window_title: str, executable_name: str) -> str:
        prompt = (
            f"Ты — эксперт по продуктивности. Твоя задача — определить, является ли текущая активность отвлекающей или продуктивной для работы, "
            f"учитывая заголовок окна и имя приложения. "
            f"Ответь ТОЛЬКО одним словом: 'Отвлечение' или 'Продуктивно'. Никаких объяснений или других слов.\n\n"
            f"Заголовок окна: '{window_title}'\n"
            f"Имя приложения: '{executable_name}'\n\n"
            f"Твой вердикт (одно слово):"  # Добавили явное указание на место ответа
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.0,
                        "num_predict": 15  # Немного увеличил для слова "продуктивно" + возможный пунктуационный знак
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            generated_text = data.get("response", "").strip().lower()
            print(f"LLM Raw Response for ('{window_title}', '{executable_name}'): '{generated_text}'")

            if "отвлечение" in generated_text:  # Ищем точное слово или его часть
                print(f"LLM: ('{window_title}', '{executable_name}') -> ОТВЛЕЧЕНИЕ (LLM)")
                return "distraction"
            elif "продуктивно" in generated_text:
                print(f"LLM: ('{window_title}', '{executable_name}') -> ПРОДУКТИВНО (LLM)")
                return "productive"
            else:
                print(
                    f"LLM не дала четкого ответа: '{generated_text}'. Заголовок: '{window_title}', Приложение: {executable_name}")
                self.status_message.emit(f"LLM неясный ответ: '{generated_text[:30]}'")

                # Fallback: Если LLM не справилась, используем ключевые слова
                lower_title = window_title.lower()
                lower_exe = executable_name.lower() if executable_name else ""

                if lower_exe:
                    for keyword in self.current_settings.get("distracting_executables", []):
                        if keyword.lower() in lower_exe:
                            print(f"LLM Fallback: '{executable_name}' -> ОТВЛЕЧЕНИЕ (Fallback by exe)")
                            return "distraction"
                for keyword in self.current_settings.get("distracting_keywords", []):
                    if keyword.lower() in lower_title:
                        print(f"LLM Fallback: '{window_title}' -> ОТВЛЕЧЕНИЕ (Fallback by title)")
                        return "distraction"

                print(f"LLM Fallback: '{window_title}' ({executable_name}) -> НЕИЗВЕСТНО (Fallback)")
                return "unknown"

        # ... (остальная часть _classify_with_ollama без изменений с обработкой ошибок) ...
        except requests.exceptions.ConnectionError:
            self.status_message.emit(f"Ошибка: Ollama не запущена на {self.ollama_url}")
            print(f"AttentionAnalyzer: Не удалось подключиться к Ollama...")
            return "ollama_down"
        except requests.exceptions.RequestException as e:
            self.status_message.emit(f"Ошибка запроса к Ollama: {e}")
            print(f"AttentionAnalyzer: Ошибка запроса к Ollama: {e}")
            return "error"
        except json.JSONDecodeError:
            self.status_message.emit("Ошибка: Неверный JSON от Ollama.")
            print("AttentionAnalyzer: Ошибка при парсинге JSON ответа от Ollama.")
            return "error"
        except Exception as e:
            self.status_message.emit(f"Неизвестная ошибка анализа: {e}")
            print(f"AttentionAnalyzer: Неизвестная ошибка в _classify_with_ollama: {e}")
            return "error"

    # Обновляем _route_action для возможного использования executable_name в сообщениях
    def _route_action(self, window_title: str, executable_name: str, classification: str):
        display_name = f"{executable_name} ({window_title})" if executable_name else window_title
        display_name_short = display_name[:70]  # Для коротких сообщений

        if classification == "ignored":
            self.status_message.emit(f"Активность: '{display_name_short}...' (Игнорируется)")
            return

        current_time = QTime.currentTime()
        start_time = QTime.fromString(self.current_settings["work_start_time"], "HH:mm")
        end_time = QTime.fromString(self.current_settings["work_end_time"], "HH:mm")

        is_work_time = False
        if start_time.isValid() and end_time.isValid():  # Проверка на валидность времени
            if start_time <= end_time:
                is_work_time = start_time <= current_time <= end_time
            else:
                is_work_time = current_time >= start_time or current_time <= end_time
        else:
            is_work_time = True  # Если время не задано или неверно, считаем рабочим временем (или другая логика)

        if not is_work_time and classification == "distraction":
            self.status_message.emit(f"Отвлечение ({display_name_short}...), но не рабочее время. Действий нет.")
            return

        if classification == "distraction":
            target_info = f"{executable_name}|{window_title}"  # Передаем и то, и другое для блокировщика
            if self.current_settings["mode"] == "soft":
                message = (f"Кажется, вы отвлекаетесь на '{display_name_short}'. "
                           f"Режим: Мягкий. Пора вернуться к работе!")
                self.action_command.emit("notify", message)
                self.status_message.emit("Отвлечение: Отправлено уведомление.")
            elif self.current_settings["mode"] == "strict":
                message = (f"Обнаружено отвлечение: '{display_name_short}'. "
                           f"Режим: Строгий. Блокирую/закрываю...")
                # Для блокировки может быть полезнее передать exe_name или более точный идентификатор
                self.action_command.emit("block_or_close", target_info)
                self.status_message.emit(f"Отвлечение ({display_name_short}): Отправлена команда блокировки.")
        elif classification == "productive":
            self.status_message.emit(f"Активность: '{display_name_short}...' (Продуктивно)")
        elif classification == "ollama_down":
            self.status_message.emit("Ollama не запущена. Мониторинг ограничен.")
        elif classification == "error":
            self.status_message.emit("Ошибка LLM. Мониторинг ограничен.")
        elif classification == "unknown":
            self.status_message.emit(f"Неизвестная активность: '{display_name_short}...'")

    def stop(self):
        self._running = False
        self.wait()
        self.status_message.emit("Анализатор внимания остановлен.")