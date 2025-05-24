import time
import cv2
import numpy as np
import pyautogui
import easyocr
from PyQt5.QtCore import QThread, pyqtSignal, QTime
import torch
from config.settings import load_settings
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import logging
from datetime import datetime


class AttentionAnalyzer(QThread):
    analysis_result = pyqtSignal(str, str)  # (window_title, classification)
    action_command = pyqtSignal(str, str)
    status_message = pyqtSignal(str)
    screenshot_analysis_result = pyqtSignal(dict)  # Новый сигнал для результатов анализа скриншотов

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.last_analysis_time = 0
        self.analysis_cooldown = 5
        self.current_settings = load_settings()

        self.window_data_queue = []
        self.processing_data = False

        # Инициализируем модель из Hugging Face
        self.classifier = None
        self.model_loaded = False
        self._initialize_model()

        # Инициализируем OCR анализатор
        self._initialize_ocr_analyzer()

        # Настройки анализа скриншотов
        self.screenshot_analysis_enabled = True
        self.last_screenshot_analysis = 0
        self.screenshot_analysis_interval = 15  # Анализируем скриншоты каждые 15 секунд
        self.screenshot_history = []

    def _initialize_model(self):
        """Инициализируем модель для классификации активности"""
        try:
            self.status_message.emit("Загрузка NLP модели...")

            # Используем более легкую модель, специально обученную для классификации
            # Эта модель хорошо понимает контекст работы и развлечений
            # model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

            # Альтернативные варианты (можно попробовать, если первая не подойдет):
            # model_name = "microsoft/DialoGPT-medium" # Для понимания контекста
            model_name = "facebook/bart-large-mnli"  # Для zero-shot классификации

            # Загружаем модель и токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Создаем pipeline для классификации
            # Для zero-shot классификации используем другой подход
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )

            self.model_loaded = True
            self.status_message.emit("NLP модель успешно загружена!")
            print("AttentionAnalyzer: Модель Hugging Face успешно инициализирована")

        except Exception as e:
            print(f"AttentionAnalyzer: Ошибка загрузки модели: {e}")
            self.status_message.emit(f"Ошибка загрузки NLP модели: {e}")
            self.model_loaded = False

    def _initialize_ocr_analyzer(self):
        """Инициализируем OCR анализатор для скриншотов"""
        try:
            self.status_message.emit("Инициализация OCR анализатора...")

            # Настройка логирования для OCR
            logging.getLogger('easyocr').setLevel(logging.WARNING)

            # Отключаем защиту от автоматизации PyAutoGUI
            pyautogui.FAILSAFE = False

            # Инициализируем EasyOCR для русского и английского языков
            self.ocr_reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())

            # Ключевые слова для определения отвлекающего контента
            self.distraction_keywords = {
                'games': [
                    'играть', 'игра', 'game', 'play', 'level', 'score', 'победа',
                    'поражение', 'restart', 'continue', 'новая игра', 'multiplayer',
                    'single player', 'steam', 'epic games', 'battle', 'fight', 'gaming'
                ],
                'entertainment': [
                    'youtube', 'ютуб', 'смотреть', 'видео', 'фильм', 'сериал',
                    'netflix', 'twitch', 'тикток', 'tiktok', 'instagram',
                    'вконтакте', 'facebook', 'подписаться', 'лайк', 'комментарий',
                    'развлечение', 'мем', 'прикол', 'funny', 'lol', 'watch', 'subscribe'
                ],
                'social_media': [
                    'чат', 'сообщение', 'друзья', 'пост', 'фото', 'селфи',
                    'stories', 'история', 'новости', 'лента', 'feed',
                    'whatsapp', 'telegram', 'discord', 'messenger', 'chat'
                ],
                'shopping': [
                    'купить', 'цена', 'скидка', 'распродажа', 'корзина',
                    'заказ', 'доставка', 'amazon', 'aliexpress', 'озон',
                    'wildberries', 'shop', 'store', 'sale', 'buy', 'cart'
                ]
            }

            # Пороговые значения для определения отвлечения
            self.distraction_threshold = 2  # Минимальное количество ключевых слов

            self.ocr_initialized = True
            self.status_message.emit("OCR анализатор готов к работе!")
            print("AttentionAnalyzer: OCR анализатор успешно инициализирован")

        except Exception as e:
            print(f"AttentionAnalyzer: Ошибка инициализации OCR: {e}")
            self.status_message.emit(f"Ошибка инициализации OCR: {e}")
            self.ocr_initialized = False

    def take_screenshot(self) -> np.ndarray:
        """Делает скриншот экрана"""
        try:
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            return cv2.resize(screenshot_np, (0, 0), fx=0.5, fy=0.5)
        except Exception as e:
            print(f"AttentionAnalyzer: Ошибка при создании скриншота: {e}")
            return None

    def extract_text_from_image(self, image: np.ndarray) -> str:
        """Извлекает текст из изображения с помощью EasyOCR"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            results = self.ocr_reader.readtext(
                threshold,
                text_threshold=0.7,
                low_text=0.4,
                link_threshold=0.4,
                blocklist="©®™•"
            )
            return ' '.join([res[1] for res in results])
        except Exception as e:
            print(f"AttentionAnalyzer: Ошибка при извлечении текста: {e}")
            return ""

    def detect_visual_elements(self, image: np.ndarray) -> List[str]:
        """Определяет визуальные элементы популярных сайтов по цветам"""
        detected_elements = []

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Проверяем характерные цвета YouTube (красный)
            red_mask = cv2.inRange(image_rgb, (200, 0, 0), (255, 50, 50))
            if np.sum(red_mask) > 10000:
                detected_elements.append('youtube_colors')

            # Проверяем синий цвет (Facebook, VK, LinkedIn)
            blue_mask = cv2.inRange(image_rgb, (0, 0, 150), (100, 100, 255))
            if np.sum(blue_mask) > 8000:
                detected_elements.append('social_media_colors')

            # Проверяем зеленый цвет (WhatsApp, некоторые игры)
            green_mask = cv2.inRange(image_rgb, (0, 150, 0), (100, 255, 100))
            if np.sum(green_mask) > 5000:
                detected_elements.append('messenger_colors')

            # Проверяем фиолетовый (Twitch, Discord)
            purple_mask = cv2.inRange(image_rgb, (100, 0, 150), (200, 100, 255))
            if np.sum(purple_mask) > 5000:
                detected_elements.append('gaming_colors')

        except Exception as e:
            print(f"AttentionAnalyzer: Ошибка анализа визуальных элементов: {e}")

        return detected_elements

    def analyze_screenshot_distraction(self, text: str, visual_elements: List[str]) -> Dict:
        """Анализирует уровень отвлекающих факторов на скриншоте"""
        distraction_score = 0
        found_keywords = []
        categories = []

        # Анализируем текст на наличие ключевых слов
        for category, keywords in self.distraction_keywords.items():
            category_score = 0
            for keyword in keywords:
                if keyword in text:
                    category_score += 1
                    found_keywords.append(keyword)

            if category_score > 0:
                distraction_score += category_score
                categories.append(category)

        # Добавляем баллы за визуальные элементы
        distraction_score += len(visual_elements)

        # Определяем уровень отвлекающих факторов
        if distraction_score >= self.distraction_threshold:
            level = "HIGH"
            is_distracted = True
        elif distraction_score >= 1:
            level = "MEDIUM"
            is_distracted = True
        else:
            level = "LOW"
            is_distracted = False

        return {
            'is_distracted': is_distracted,
            'distraction_level': level,
            'score': distraction_score,
            'found_keywords': found_keywords,
            'categories': categories,
            'visual_elements': visual_elements,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def analyze_current_screen(self) -> Dict:
        """Анализирует текущий экран на предмет отвлекающих факторов"""
        if not self.ocr_initialized:
            return {'error': 'OCR не инициализирован'}

        try:
            # Делаем скриншот
            image = self.take_screenshot()
            if image is None:
                return {'error': 'Не удалось сделать скриншот'}

            # Извлекаем текст
            text = self.extract_text_from_image(image)

            # Анализируем визуальные элементы
            visual_elements = self.detect_visual_elements(image)

            # Анализируем уровень отвлечения
            analysis = self.analyze_screenshot_distraction(text, visual_elements)

            # Сохраняем в историю
            self.screenshot_history.append(analysis)

            return analysis

        except Exception as e:
            print(f"AttentionAnalyzer: Ошибка анализа скриншота: {e}")
            return {'error': f'Ошибка анализа: {str(e)}'}

    def update_settings(self):
        self.current_settings = load_settings()

        # Обновляем настройки анализа скриншотов из конфигурации
        if 'screenshot_analysis_enabled' in self.current_settings:
            self.screenshot_analysis_enabled = self.current_settings['screenshot_analysis_enabled']
        if 'screenshot_analysis_interval' in self.current_settings:
            self.screenshot_analysis_interval = self.current_settings['screenshot_analysis_interval']
        if 'distraction_threshold' in self.current_settings:
            self.distraction_threshold = self.current_settings['distraction_threshold']

        print("AttentionAnalyzer: Настройки обновлены.")

    def add_data_for_analysis(self, window_title: str, executable_name: str):
        current_data = (window_title or "", executable_name or "")

        if not current_data[0] and not current_data[1]:
            return

        if not self.window_data_queue or self.window_data_queue[-1] != current_data:
            self.window_data_queue.append(current_data)

    def run(self):
        self.status_message.emit("Анализатор внимания запущен. Ожидание данных...")

        while self._running:
            current_time = time.time()

            # Анализ окон
            if self.window_data_queue and not self.processing_data:
                if current_time - self.last_analysis_time >= self.analysis_cooldown:
                    data_to_analyze = self.window_data_queue.pop()
                    self.window_data_queue.clear()

                    title_to_analyze, exe_to_analyze = data_to_analyze
                    self.processing_data = True
                    self.status_message.emit(f"Анализ: '{title_to_analyze[:30]}...' ({exe_to_analyze})")

                    classification = self._pre_classify_with_keywords(title_to_analyze, exe_to_analyze)

                    if classification == "needs_llm":
                        classification = self._classify_with_huggingface(title_to_analyze, exe_to_analyze)

                    self.analysis_result.emit(title_to_analyze, classification)
                    self._route_action(title_to_analyze, exe_to_analyze, classification)

                    self.last_analysis_time = time.time()
                    self.processing_data = False

            # Анализ скриншотов
            if (self.screenshot_analysis_enabled and
                    current_time - self.last_screenshot_analysis >= self.screenshot_analysis_interval):

                self.status_message.emit("Анализ скриншота экрана...")
                screenshot_analysis = self.analyze_current_screen()

                if 'error' not in screenshot_analysis:
                    self.screenshot_analysis_result.emit(screenshot_analysis)

                    # Если обнаружено отвлечение на скриншоте, обрабатываем его
                    if screenshot_analysis.get('is_distracted', False):
                        self._handle_screenshot_distraction(screenshot_analysis)

                self.last_screenshot_analysis = current_time

            time.sleep(0.5)

    def _handle_screenshot_distraction(self, analysis: Dict):
        """Обрабатывает обнаруженное отвлечение на скриншоте"""
        try:
            level = analysis.get('distraction_level', 'UNKNOWN')
            keywords = analysis.get('found_keywords', [])
            categories = analysis.get('categories', [])

            # Проверяем рабочее время
            current_time = QTime.currentTime()
            start_time = QTime.fromString(self.current_settings["work_start_time"], "HH:mm")
            end_time = QTime.fromString(self.current_settings["work_end_time"], "HH:mm")

            is_work_time = False
            if start_time.isValid() and end_time.isValid():
                if start_time <= end_time:
                    is_work_time = start_time <= current_time <= end_time
                else:
                    is_work_time = current_time >= start_time or current_time <= end_time
            else:
                is_work_time = True

            if not is_work_time:
                self.status_message.emit(f"Отвлечение обнаружено на экране ({level}), но не рабочее время.")
                return

            # Формируем сообщение о найденном отвлечении
            distraction_info = f"Уровень: {level}, Категории: {', '.join(categories)}"
            if keywords:
                distraction_info += f", Ключевые слова: {', '.join(keywords[:5])}"  # Первые 5 слов

            if self.current_settings["mode"] == "soft":
                message = (f"Обнаружено отвлечение на экране! {distraction_info}. "
                           f"Режим: Мягкий. Время вернуться к работе!")
                self.action_command.emit("notify", message)
                self.status_message.emit(f"Отвлечение на экране ({level}): Отправлено уведомление.")
            elif self.current_settings["mode"] == "strict":
                # В строгом режиме можем закрыть активное окно или заблокировать доступ
                message = (f"Обнаружено отвлечение на экране! {distraction_info}. "
                           f"Режим: Строгий. Принимаю меры...")
                # Для скриншотов сложнее определить конкретное приложение для блокировки
                # Поэтому отправляем общее уведомление и команду на закрытие активного окна
                self.action_command.emit("close_active_window", "")
                self.status_message.emit(f"Отвлечение на экране ({level}): Закрытие активного окна.")

        except Exception as e:
            print(f"AttentionAnalyzer: Ошибка обработки отвлечения на скриншоте: {e}")

    def get_screenshot_statistics(self) -> Dict:
        """Возвращает статистику по анализу скриншотов"""
        if not self.screenshot_history:
            return {'message': 'Нет данных анализа скриншотов'}

        total_screenshots = len(self.screenshot_history)
        distracted_screenshots = sum(1 for s in self.screenshot_history if s.get('is_distracted', False))

        # Подсчитываем категории
        category_stats = {}
        for screenshot in self.screenshot_history:
            for category in screenshot.get('categories', []):
                category_stats[category] = category_stats.get(category, 0) + 1

        return {
            'total_screenshots_analyzed': total_screenshots,
            'distracted_screenshots': distracted_screenshots,
            'distraction_percentage': (
                                                  distracted_screenshots / total_screenshots) * 100 if total_screenshots > 0 else 0,
            'most_common_distractions': category_stats,
            'average_score': np.mean(
                [s.get('score', 0) for s in self.screenshot_history]) if self.screenshot_history else 0
        }

    def _pre_classify_with_keywords(self, window_title: str, executable_name: str) -> str:
        lower_title = window_title.lower()
        lower_exe = executable_name.lower() if executable_name else ""

        # Игнорируемые
        if lower_exe:
            for keyword in self.current_settings.get("ignored_executables", []):
                if keyword.lower() in lower_exe:
                    print(f"AttentionAnalyzer: '{executable_name}' классифицирован как IGNORED по exe.")
                    return "ignored"
        for keyword in self.current_settings.get("ignored_keywords", []):
            if keyword.lower() in lower_title:
                print(f"AttentionAnalyzer: '{window_title}' классифицирован как IGNORED по заголовку.")
                return "ignored"

        # Продуктивные
        if lower_exe:
            for keyword in self.current_settings.get("productive_executables", []):
                if keyword.lower() in lower_exe:
                    print(f"AttentionAnalyzer: '{executable_name}' классифицирован как PRODUCTIVE по exe.")
                    return "productive"
        for keyword in self.current_settings.get("productive_keywords", []):
            if keyword.lower() in lower_title:
                print(f"AttentionAnalyzer: '{window_title}' классифицирован как PRODUCTIVE по заголовку.")
                return "productive"

        # Отвлекающие
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

    def _classify_with_huggingface(self, window_title: str, executable_name: str) -> str:
        """Классифицируем активность с помощью модели Hugging Face"""

        if not self.model_loaded:
            print("AttentionAnalyzer: Модель не загружена, используем fallback")
            return self._fallback_classification(window_title, executable_name)

        try:
            # Формируем текст для анализа
            text_to_analyze = f"{window_title} {executable_name}".strip()
            if not text_to_analyze:
                return "unknown"

            # Определяем категории для zero-shot классификации
            candidate_labels = [
                "work and productivity",
                "entertainment and distraction",
                "social media and messaging",
                "programming and development",
                "research and learning",
                "gaming and fun"
            ]

            # Выполняем классификацию
            result = self.classifier(text_to_analyze, candidate_labels)

            # Получаем наиболее вероятную категорию
            top_label = result['labels'][0]
            confidence = result['scores'][0]

            print(f"HuggingFace Classification: '{text_to_analyze}' -> {top_label} (confidence: {confidence:.3f})")

            # Сопоставляем результаты с нашими категориями
            if confidence < 0.6:  # Низкая уверенность
                print(f"HuggingFace: Низкая уверенность ({confidence:.3f}), используем fallback")
                return self._fallback_classification(window_title, executable_name)

            # Маппинг категорий на наши классификации
            if top_label in ["work and productivity", "programming and development", "research and learning"]:
                print(f"HuggingFace: '{window_title}' ({executable_name}) -> ПРОДУКТИВНО")
                return "productive"
            elif top_label in ["entertainment and distraction", "social media and messaging", "gaming and fun"]:
                print(f"HuggingFace: '{window_title}' ({executable_name}) -> ОТВЛЕЧЕНИЕ")
                return "distraction"
            else:
                print(f"HuggingFace: Неопределенная категория '{top_label}', используем fallback")
                return self._fallback_classification(window_title, executable_name)

        except Exception as e:
            print(f"AttentionAnalyzer: Ошибка при классификации с HuggingFace: {e}")
            self.status_message.emit(f"Ошибка NLP: {str(e)[:50]}...")
            return self._fallback_classification(window_title, executable_name)

    def _fallback_classification(self, window_title: str, executable_name: str) -> str:
        """Резервная классификация на основе простых правил"""
        lower_title = window_title.lower()
        lower_exe = executable_name.lower() if executable_name else ""

        # Простые эвристики для fallback
        work_indicators = [
            'code', 'visual studio', 'pycharm', 'github', 'git', 'terminal', 'cmd', 'powershell',
            'excel', 'word', 'docs', 'sheets', 'notion', 'jira', 'confluence', 'slack',
            'zoom', 'teams', 'email', 'outlook', 'calendar'
        ]

        distraction_indicators = [
            'youtube', 'netflix', 'instagram', 'facebook', 'twitter', 'tiktok', 'reddit',
            'gaming', 'steam', 'game', 'play', 'entertainment', 'music', 'spotify',
            'whatsapp', 'telegram', 'discord', 'chat'
        ]

        text_combined = f"{lower_title} {lower_exe}"

        work_score = sum(1 for indicator in work_indicators if indicator in text_combined)
        distraction_score = sum(1 for indicator in distraction_indicators if indicator in text_combined)

        if work_score > distraction_score and work_score > 0:
            print(f"Fallback: '{window_title}' -> ПРОДУКТИВНО (score: {work_score})")
            return "productive"
        elif distraction_score > 0:
            print(f"Fallback: '{window_title}' -> ОТВЛЕЧЕНИЕ (score: {distraction_score})")
            return "distraction"
        else:
            print(f"Fallback: '{window_title}' -> НЕИЗВЕСТНО")
            return "unknown"

    def _route_action(self, window_title: str, executable_name: str, classification: str):
        display_name = f"{executable_name} ({window_title})" if executable_name else window_title
        display_name_short = display_name[:70]

        if classification == "ignored":
            self.status_message.emit(f"Активность: '{display_name_short}...' (Игнорируется)")
            return

        current_time = QTime.currentTime()
        start_time = QTime.fromString(self.current_settings["work_start_time"], "HH:mm")
        end_time = QTime.fromString(self.current_settings["work_end_time"], "HH:mm")

        is_work_time = False
        if start_time.isValid() and end_time.isValid():
            if start_time <= end_time:
                is_work_time = start_time <= current_time <= end_time
            else:
                is_work_time = current_time >= start_time or current_time <= end_time
        else:
            is_work_time = True

        if not is_work_time and classification == "distraction":
            self.status_message.emit(f"Отвлечение ({display_name_short}...), но не рабочее время. Действий нет.")
            return

        if classification == "distraction":
            target_info = f"{executable_name}|{window_title}"
            if self.current_settings["mode"] == "soft":
                message = (f"Кажется, вы отвлекаетесь на '{display_name_short}'. "
                           f"Режим: Мягкий. Пора вернуться к работе!")
                self.action_command.emit("notify", message)
                self.status_message.emit("Отвлечение: Отправлено уведомление.")
            elif self.current_settings["mode"] == "strict":
                message = (f"Обнаружено отвлечение: '{display_name_short}'. "
                           f"Режим: Строгий. Блокирую/закрываю...")
                self.action_command.emit("block_or_close", target_info)
                self.status_message.emit(f"Отвлечение ({display_name_short}): Отправлена команда блокировки.")
        elif classification == "productive":
            self.status_message.emit(f"Активность: '{display_name_short}...' (Продуктивно)")
        elif classification == "model_error":
            self.status_message.emit("Ошибка NLP модели. Мониторинг ограничен.")
        elif classification == "unknown":
            self.status_message.emit(f"Неизвестная активность: '{display_name_short}...'")

    def stop(self):
        self._running = False
        self.wait()
        self.status_message.emit("Анализатор внимания остановлен.")