import time
import cv2
import numpy as np
import pyautogui
import easyocr
from PyQt5.QtCore import QThread, pyqtSignal, QTime
import torch
from config.settings import load_settings
from transformers import pipeline
from typing import List, Dict, Optional, Callable, Any
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import json


class TaskType(Enum):
    """Типы задач для маршрутизации"""
    WINDOW_ANALYSIS = "window_analysis"
    SCREENSHOT_ANALYSIS = "screenshot_analysis"
    KEYWORD_CLASSIFICATION = "keyword_classification"
    NLP_CLASSIFICATION = "nlp_classification"
    VISUAL_DETECTION = "visual_detection"
    OCR_EXTRACTION = "ocr_extraction"


class Priority(Enum):
    """Приоритеты задач"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AnalysisTask:
    """Задача для анализа"""
    task_id: str
    task_type: TaskType
    priority: Priority
    data: Dict[str, Any]
    callback: Optional[Callable] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class AnalysisResult:
    """Результат анализа"""
    task_id: str
    task_type: TaskType
    success: bool
    result: Any
    error: Optional[str] = None
    processing_time: float = 0
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


class TaskRouter:
    """Маршрутизатор задач - распределяет задачи между специализированными агентами"""

    def __init__(self):
        self.agents = {}
        self.task_queue = []
        self.processing_stats = {}

    def register_agent(self, task_type: TaskType, agent: 'BaseAgent'):
        """Регистрирует агента для определенного типа задач"""
        self.agents[task_type] = agent
        self.processing_stats[task_type] = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'average_time': 0,
            'last_performance': 1.0
        }

    def route_task(self, task: AnalysisTask) -> Optional['BaseAgent']:
        """Маршрутизирует задачу к подходящему агенту"""
        if task.task_type in self.agents:
            agent = self.agents[task.task_type]
            if agent.can_handle_task(task):
                return agent

        # Fallback логика - ищем агента, который может обработать задачу
        for agent in self.agents.values():
            if agent.can_handle_task(task):
                return agent

        return None

    def update_performance_stats(self, result: AnalysisResult):
        """Обновляет статистику производительности агентов"""
        task_type = result.task_type
        if task_type in self.processing_stats:
            stats = self.processing_stats[task_type]
            stats['total_tasks'] += 1
            if result.success:
                stats['successful_tasks'] += 1

            # Обновляем среднее время обработки
            if stats['total_tasks'] == 1:
                stats['average_time'] = result.processing_time
            else:
                stats['average_time'] = (stats['average_time'] * (stats['total_tasks'] - 1) +
                                         result.processing_time) / stats['total_tasks']

            # Вычисляем производительность
            success_rate = stats['successful_tasks'] / stats['total_tasks']
            time_efficiency = max(0.1, 1.0 - min(1.0, stats['average_time'] / 10.0))  # Нормализуем время
            stats['last_performance'] = (success_rate * 0.7 + time_efficiency * 0.3)


class BaseAgent:
    """Базовый класс для всех агентов"""

    def __init__(self, name: str):
        self.name = name
        self.supported_tasks = set()
        self.performance_history = []
        self.is_busy = False

    def can_handle_task(self, task: AnalysisTask) -> bool:
        """Проверяет, может ли агент обработать задачу"""
        return task.task_type in self.supported_tasks and not self.is_busy

    async def process_task(self, task: AnalysisTask) -> AnalysisResult:
        """Обрабатывает задачу асинхронно"""
        start_time = time.time()
        self.is_busy = True

        try:
            result = await self._execute_task(task)
            processing_time = time.time() - start_time

            return AnalysisResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=True,
                result=result,
                processing_time=processing_time,
                confidence=getattr(result, 'confidence', 1.0) if hasattr(result, 'confidence') else 1.0
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return AnalysisResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=False,
                result=None,
                error=str(e),
                processing_time=processing_time
            )
        finally:
            self.is_busy = False

    async def _execute_task(self, task: AnalysisTask) -> Any:
        """Переопределяется в наследниках для выполнения конкретной задачи"""
        raise NotImplementedError


class KeywordClassificationAgent(BaseAgent):
    """Агент для быстрой классификации по ключевым словам"""

    def __init__(self, settings: Dict):
        super().__init__("KeywordClassifier")
        self.supported_tasks = {TaskType.KEYWORD_CLASSIFICATION}
        self.settings = settings

    async def _execute_task(self, task: AnalysisTask) -> Dict:
        """Выполняет классификацию по ключевым словам"""
        window_title = task.data.get('window_title', '').lower()
        executable_name = task.data.get('executable_name', '').lower()

        # Проверяем игнорируемые
        for keyword in self.settings.get("ignored_keywords", []):
            if keyword.lower() in window_title:
                return {"classification": "ignored", "confidence": 1.0, "matched_keyword": keyword}

        for keyword in self.settings.get("ignored_executables", []):
            if keyword.lower() in executable_name:
                return {"classification": "ignored", "confidence": 1.0, "matched_keyword": keyword}

        # Проверяем продуктивные
        for keyword in self.settings.get("productive_keywords", []):
            if keyword.lower() in window_title:
                return {"classification": "productive", "confidence": 0.9, "matched_keyword": keyword}

        for keyword in self.settings.get("productive_executables", []):
            if keyword.lower() in executable_name:
                return {"classification": "productive", "confidence": 0.9, "matched_keyword": keyword}

        # Проверяем отвлекающие
        for keyword in self.settings.get("distracting_keywords", []):
            if keyword.lower() in window_title:
                return {"classification": "distraction", "confidence": 0.9, "matched_keyword": keyword}

        for keyword in self.settings.get("distracting_executables", []):
            if keyword.lower() in executable_name:
                return {"classification": "distraction", "confidence": 0.9, "matched_keyword": keyword}

        return {"classification": "unknown", "confidence": 0.0, "matched_keyword": None}


class NLPClassificationAgent(BaseAgent):
    """Агент для NLP-классификации с помощью Hugging Face"""

    def __init__(self):
        super().__init__("NLPClassifier")
        self.supported_tasks = {TaskType.NLP_CLASSIFICATION}
        self.classifier = None
        self.model_loaded = False
        self._initialize_model()

    def _initialize_model(self):
        """Инициализирует NLP модель"""
        try:
            model_name = "facebook/bart-large-mnli"
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_loaded = True
            print("NLPClassificationAgent: Модель успешно загружена")
        except Exception as e:
            print(f"NLPClassificationAgent: Ошибка загрузки модели: {e}")
            self.model_loaded = False

    def can_handle_task(self, task: AnalysisTask) -> bool:
        return super().can_handle_task(task) and self.model_loaded

    async def _execute_task(self, task: AnalysisTask) -> Dict:
        """Выполняет NLP классификацию"""
        window_title = task.data.get('window_title', '')
        executable_name = task.data.get('executable_name', '')

        text_to_analyze = f"{window_title} {executable_name}".strip()
        if not text_to_analyze:
            return {"classification": "unknown", "confidence": 0.0}

        candidate_labels = [
            "work and productivity",
            "entertainment and distraction",
            "social media and messaging",
            "programming and development",
            "research and learning",
            "gaming and fun"
        ]

        # Симуляция асинхронности для тяжелой операции
        await asyncio.sleep(0.01)

        result = self.classifier(text_to_analyze, candidate_labels)
        top_label = result['labels'][0]
        confidence = result['scores'][0]

        # Маппинг на наши категории
        if top_label in ["work and productivity", "programming and development", "research and learning"]:
            classification = "productive"
        elif top_label in ["entertainment and distraction", "social media and messaging", "gaming and fun"]:
            classification = "distraction"
        else:
            classification = "unknown"

        return {
            "classification": classification,
            "confidence": confidence,
            "top_label": top_label,
            "all_results": result
        }


class OCRAgent(BaseAgent):
    """Агент для извлечения текста из изображений"""

    def __init__(self):
        super().__init__("OCRExtractor")
        self.supported_tasks = {TaskType.OCR_EXTRACTION}
        self.ocr_reader = None
        self._initialize_ocr()

    def _initialize_ocr(self):
        """Инициализирует OCR"""
        try:
            logging.getLogger('easyocr').setLevel(logging.WARNING)
            self.ocr_reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
            print("OCRAgent: OCR успешно инициализирован")
        except Exception as e:
            print(f"OCRAgent: Ошибка инициализации OCR: {e}")

    def can_handle_task(self, task: AnalysisTask) -> bool:
        return super().can_handle_task(task) and self.ocr_reader is not None

    async def _execute_task(self, task: AnalysisTask) -> Dict:
        """Извлекает текст из изображения"""
        image = task.data.get('image')
        if image is None:
            raise ValueError("Изображение не предоставлено")

        # Симуляция асинхронности
        await asyncio.sleep(0.01)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        results = self.ocr_reader.readtext(
            threshold,
            text_threshold=0.7,
            low_text=0.4,
            link_threshold=0.4,
            blocklist="©®™•"
        )

        extracted_text = ' '.join([res[1] for res in results])

        return {
            "extracted_text": extracted_text,
            "raw_results": results,
            "confidence": np.mean([res[2] for res in results]) if results else 0.0
        }


class VisualDetectionAgent(BaseAgent):
    """Агент для обнаружения визуальных элементов"""

    def __init__(self):
        super().__init__("VisualDetector")
        self.supported_tasks = {TaskType.VISUAL_DETECTION}

    async def _execute_task(self, task: AnalysisTask) -> Dict:
        """Обнаруживает визуальные элементы на изображении"""
        image = task.data.get('image')
        if image is None:
            raise ValueError("Изображение не предоставлено")

        detected_elements = []
        confidence_scores = {}

        # Симуляция асинхронности
        await asyncio.sleep(0.01)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Анализ цветов
        color_detections = [
            ('youtube_colors', (200, 0, 0), (255, 50, 50), 10000),
            ('social_media_colors', (0, 0, 150), (100, 100, 255), 8000),
            ('messenger_colors', (0, 150, 0), (100, 255, 100), 5000),
            ('gaming_colors', (100, 0, 150), (200, 100, 255), 5000)
        ]

        for element_name, lower_bound, upper_bound, threshold in color_detections:
            mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
            pixel_count = np.sum(mask)

            if pixel_count > threshold:
                detected_elements.append(element_name)
                confidence_scores[element_name] = min(1.0, pixel_count / (threshold * 2))

        return {
            "detected_elements": detected_elements,
            "confidence_scores": confidence_scores,
            "total_elements": len(detected_elements)
        }


class PerformanceEvaluator:
    """Evaluator для оценки производительности системы"""

    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {
            'accuracy': 0.0,
            'response_time': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0
        }

    def evaluate_result(self, result: AnalysisResult, expected_result: Optional[Any] = None) -> Dict:
        """Оценивает качество результата"""
        evaluation = {
            'timestamp': time.time(),
            'task_type': result.task_type.value,
            'success': result.success,
            'processing_time': result.processing_time,
            'confidence': result.confidence
        }

        # Оценка точности (если есть ожидаемый результат)
        if expected_result is not None and result.success:
            accuracy = self._calculate_accuracy(result.result, expected_result)
            evaluation['accuracy'] = accuracy

        # Оценка скорости
        speed_score = self._evaluate_speed(result.processing_time, result.task_type)
        evaluation['speed_score'] = speed_score

        self.metrics_history.append(evaluation)
        self._update_current_metrics()

        return evaluation

    def _calculate_accuracy(self, actual: Any, expected: Any) -> float:
        """Вычисляет точность результата"""
        if isinstance(actual, dict) and isinstance(expected, dict):
            if 'classification' in actual and 'classification' in expected:
                return 1.0 if actual['classification'] == expected['classification'] else 0.0
        return 0.5  # Default для неопределенных случаев

    def _evaluate_speed(self, processing_time: float, task_type: TaskType) -> float:
        """Оценивает скорость обработки"""
        # Целевые времена для разных типов задач (в секундах)
        target_times = {
            TaskType.KEYWORD_CLASSIFICATION: 0.01,
            TaskType.NLP_CLASSIFICATION: 0.5,
            TaskType.OCR_EXTRACTION: 1.0,
            TaskType.VISUAL_DETECTION: 0.2,
            TaskType.SCREENSHOT_ANALYSIS: 2.0,
            TaskType.WINDOW_ANALYSIS: 0.1
        }

        target_time = target_times.get(task_type, 1.0)
        speed_score = max(0.0, 1.0 - (processing_time / target_time))
        return min(1.0, speed_score)

    def _update_current_metrics(self):
        """Обновляет текущие метрики на основе истории"""
        if not self.metrics_history:
            return

        recent_results = self.metrics_history[-50:]  # Последние 50 результатов

        # Вычисляем средние значения
        total_results = len(recent_results)
        successful_results = sum(1 for r in recent_results if r['success'])

        self.current_metrics['accuracy'] = successful_results / total_results
        self.current_metrics['response_time'] = np.mean([r['processing_time'] for r in recent_results])  # type: ignore
        self.current_metrics['error_rate'] = 1.0 - (successful_results / total_results)

    def get_performance_report(self) -> Dict:
        """Возвращает отчет о производительности"""
        return {
            'current_metrics': self.current_metrics,
            'total_evaluations': len(self.metrics_history),
            'evaluation_history': self.metrics_history[-10:],  # Последние 10
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Генерирует рекомендации по улучшению"""
        recommendations = []

        if self.current_metrics['accuracy'] < 0.8:
            recommendations.append("Рассмотреть улучшение алгоритмов классификации")

        if self.current_metrics['response_time'] > 2.0:
            recommendations.append("Оптимизировать время обработки задач")

        if self.current_metrics['error_rate'] > 0.1:
            recommendations.append("Улучшить обработку ошибок и стабильность")

        return recommendations


class SystemOptimizer:
    """Optimizer для оптимизации параметров системы"""

    def __init__(self, router: TaskRouter, evaluator: PerformanceEvaluator):
        self.router = router
        self.evaluator = evaluator
        self.optimization_history = []
        self.current_config = {
            'analysis_cooldown': 5,
            'screenshot_interval': 20,
            'distraction_threshold': 2,
            'max_concurrent_tasks': 3
        }

    def optimize_parameters(self) -> Dict:
        """Оптимизирует параметры системы на основе производительности"""
        performance_report = self.evaluator.get_performance_report()
        current_metrics = performance_report['current_metrics']

        optimization_result = {
            'timestamp': time.time(),
            'previous_config': self.current_config.copy(),
            'performance_before': current_metrics.copy(),
            'changes_made': []
        }

        # Оптимизация на основе времени отклика
        if current_metrics['response_time'] > 1.5:
            # Увеличиваем интервалы для снижения нагрузки
            if self.current_config['analysis_cooldown'] < 10:
                self.current_config['analysis_cooldown'] += 1
                optimization_result['changes_made'].append("Увеличен интервал анализа окон")

            if self.current_config['screenshot_interval'] < 60:
                self.current_config['screenshot_interval'] += 5
                optimization_result['changes_made'].append("Увеличен интервал скриншотов")

        elif current_metrics['response_time'] < 0.5 and current_metrics['accuracy'] > 0.9:
            # Система работает хорошо, можно увеличить частоту анализа
            if self.current_config['analysis_cooldown'] > 2:
                self.current_config['analysis_cooldown'] -= 1
                optimization_result['changes_made'].append("Уменьшен интервал анализа окон")

        # Оптимизация точности
        if current_metrics['accuracy'] < 0.7:
            if self.current_config['distraction_threshold'] > 1:
                self.current_config['distraction_threshold'] -= 1
                optimization_result['changes_made'].append("Снижен порог отвлечения для большей чувствительности")

        self.optimization_history.append(optimization_result)
        return optimization_result

    def get_optimized_config(self) -> Dict:
        """Возвращает оптимизированную конфигурацию"""
        return self.current_config.copy()


class AttentionAnalyzer(QThread):
    """Улучшенный анализатор внимания с паттернами Agent Recipes"""

    analysis_result = pyqtSignal(str, str)
    action_command = pyqtSignal(str, str)
    status_message = pyqtSignal(str)
    screenshot_analysis_result = pyqtSignal(dict)
    performance_report = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.current_settings = load_settings()

        # Инициализация компонентов Agent Recipes
        self.task_router = TaskRouter()
        self.evaluator = PerformanceEvaluator()
        self.optimizer = SystemOptimizer(self.task_router, self.evaluator)

        # Пул задач и executor для параллелизма
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Инициализация агентов
        self._initialize_agents()

        # Данные для анализа
        self.window_data_queue = []
        self.screenshot_history = []
        self.last_analysis_time = 0
        self.last_screenshot_analysis = 0
        self.last_optimization = 0

        # Получаем оптимизированные параметры
        self.optimized_config = self.optimizer.get_optimized_config()

    def _initialize_agents(self):
        """Инициализирует всех агентов"""
        try:
            # Регистрируем агентов
            keyword_agent = KeywordClassificationAgent(self.current_settings)
            nlp_agent = NLPClassificationAgent()
            ocr_agent = OCRAgent()
            visual_agent = VisualDetectionAgent()

            self.task_router.register_agent(TaskType.KEYWORD_CLASSIFICATION, keyword_agent)
            self.task_router.register_agent(TaskType.NLP_CLASSIFICATION, nlp_agent)
            self.task_router.register_agent(TaskType.OCR_EXTRACTION, ocr_agent)
            self.task_router.register_agent(TaskType.VISUAL_DETECTION, visual_agent)

            self.status_message.emit("Все агенты успешно инициализированы")  # type: ignore

        except Exception as e:
            print(f"Ошибка инициализации агентов: {e}")
            self.status_message.emit(f"Ошибка инициализации: {e}")  # type: ignore

    async def process_task_async(self, task: AnalysisTask) -> AnalysisResult:
        """Асинхронно обрабатывает задачу"""
        agent = self.task_router.route_task(task)
        if agent is None:
            return AnalysisResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=False,
                error="Агент не найден"
            )

        result = await agent.process_task(task)

        # Оценка результата
        evaluation = self.evaluator.evaluate_result(result)
        self.task_router.update_performance_stats(result)

        return result

    def add_data_for_analysis(self, window_title: str, executable_name: str):
        """Добавляет данные для анализа"""
        current_data = (window_title or "", executable_name or "")

        if not current_data[0] and not current_data[1]:
            return

        if not self.window_data_queue or self.window_data_queue[-1] != current_data:
            self.window_data_queue.append(current_data)

    def take_screenshot(self) -> Optional[np.ndarray]:
        """Делает скриншот экрана"""
        try:
            pyautogui.FAILSAFE = False
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            return cv2.resize(screenshot_np, (0, 0), fx=0.5, fy=0.5)
        except Exception as e:
            print(f"Ошибка при создании скриншота: {e}")
            return None

    async def analyze_window_async(self, window_title: str, executable_name: str) -> str:
        """Асинхронно анализирует окно"""
        # Сначала пробуем быструю классификацию по ключевым словам
        keyword_task = AnalysisTask(
            task_id=f"keyword_{time.time()}",
            task_type=TaskType.KEYWORD_CLASSIFICATION,
            priority=Priority.HIGH,
            data={'window_title': window_title, 'executable_name': executable_name}
        )

        keyword_result = await self.process_task_async(keyword_task)

        if (keyword_result.success and
                keyword_result.result.get('classification') != 'unknown' and
                keyword_result.result.get('confidence', 0) > 0.8):
            return keyword_result.result['classification']

        # Если ключевые слова не дали результата, используем NLP
        nlp_task = AnalysisTask(
            task_id=f"nlp_{time.time()}",
            task_type=TaskType.NLP_CLASSIFICATION,
            priority=Priority.MEDIUM,
            data={'window_title': window_title, 'executable_name': executable_name}
        )

        nlp_result = await self.process_task_async(nlp_task)

        if nlp_result.success:
            return nlp_result.result.get('classification', 'unknown')

        return 'unknown'

    async def analyze_screenshot_async(self) -> Optional[Dict]:
        """Асинхронно анализирует скриншот"""
        image = self.take_screenshot()
        if image is None:
            return None

        # Создаем задачи для параллельного выполнения
        tasks = []

        # OCR задача
        ocr_task = AnalysisTask(
            task_id=f"ocr_{time.time()}",
            task_type=TaskType.OCR_EXTRACTION,
            priority=Priority.MEDIUM,
            data={'image': image}
        )
        tasks.append(self.process_task_async(ocr_task))

        # Визуальный анализ
        visual_task = AnalysisTask(
            task_id=f"visual_{time.time()}",
            task_type=TaskType.VISUAL_DETECTION,
            priority=Priority.MEDIUM,
            data={'image': image}
        )
        tasks.append(self.process_task_async(visual_task))

        # Выполняем задачи параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обрабатываем результаты
        extracted_text = ""
        visual_elements = []

        for result in results:
            if isinstance(result, AnalysisResult) and result.success:
                if result.task_type == TaskType.OCR_EXTRACTION:
                    extracted_text = result.result.get('extracted_text', '')
                elif result.task_type == TaskType.VISUAL_DETECTION:
                    visual_elements = result.result.get('detected_elements', [])

        # Анализ отвлечения
        return self._analyze_distraction(extracted_text, visual_elements)

    def _analyze_distraction(self, text: str, visual_elements: List[str]) -> Dict:
        """Анализирует уровень отвлечения"""
        distraction_keywords = {
            'games': ['играть', 'игра', 'game', 'play', 'level', 'score', 'steam'],
            'entertainment': ['youtube', 'видео', 'фильм', 'netflix', 'twitch'],
            'social_media': ['чат', 'сообщение', 'пост', 'лайк', 'instagram', 'facebook'],
            'shopping': ['купить', 'цена', 'скидка', 'корзина', 'amazon', 'shop']
        }

        distraction_score = 0
        found_keywords = []
        categories = []

        text_lower = text.lower()

        # Анализируем текст на наличие ключевых слов
        for category, keywords in distraction_keywords.items():
            category_score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    category_score += 1
                    found_keywords.append(keyword)

            if category_score > 0:
                distraction_score += category_score
                categories.append(category)

        # Добавляем баллы за визуальные элементы
        distraction_score += len(visual_elements)

        # Определяем уровень отвлечения
        threshold = self.optimized_config.get('distraction_threshold', 2)
        if distraction_score >= threshold:
            level = "HIGH"
            is_distracted = True
        elif distraction_score >= 1:
            level = "MEDIUM"
            is_distracted = True
        else:
            level = "LOW"
            is_distracted = False

        analysis_result1 = {
            'is_distracted': is_distracted,
            'distraction_level': level,
            'score': distraction_score,
            'found_keywords': found_keywords,
            'categories': categories,
            'visual_elements': visual_elements,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Сохраняем в историю
        self.screenshot_history.append(analysis_result1)
        return analysis_result1

    def run(self):
        """Основной цикл работы анализатора"""
        self.status_message.emit("Улучшенный анализатор внимания запущен...")  # type: ignore

        # Создаем event loop для async операций
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self._running:
                current_time = time.time()

                # Анализ окон
                if self.window_data_queue:
                    cooldown = self.optimized_config.get('analysis_cooldown', 5)
                    if current_time - self.last_analysis_time >= cooldown:
                        data_to_analyze = self.window_data_queue.pop()
                        self.window_data_queue.clear()

                        title, exe = data_to_analyze
                        self.status_message.emit(f"Анализ: '{title[:30]}...' ({exe})")  # type: ignore

                        # Асинхронный анализ окна
                        try:
                            classification = loop.run_until_complete(
                                self.analyze_window_async(title, exe)
                            )

                            self.analysis_result.emit(title, classification)  # type: ignore
                            self._route_action(title, exe, classification)
                            self.last_analysis_time = current_time
                        except Exception as e:
                            print(f"Ошибка анализа окна: {e}")
                            self.status_message.emit(f"Ошибка анализа: {str(e)[:50]}...")  # type: ignore

                # Анализ скриншотов
                screenshot_interval = self.optimized_config.get('screenshot_interval', 20)
                if current_time - self.last_screenshot_analysis >= screenshot_interval:
                    self.status_message.emit("Анализ скриншота экрана...")  # type: ignore

                    try:
                        screenshot_analysis = loop.run_until_complete(
                            self.analyze_screenshot_async()
                        )

                        if screenshot_analysis:
                            self.screenshot_analysis_result.emit(screenshot_analysis)  # type: ignore

                            if screenshot_analysis.get('is_distracted', False):
                                self._handle_screenshot_distraction(screenshot_analysis)

                        self.last_screenshot_analysis = current_time
                    except Exception as e:
                        print(f"Ошибка анализа скриншота: {e}")
                        self.status_message.emit(f"Ошибка скриншота: {str(e)[:50]}...")  # type: ignore

                # Оптимизация системы (каждые 5 минут)
                if current_time - self.last_optimization >= 300:
                    try:
                        optimization_result = self.optimizer.optimize_parameters()
                        self.optimized_config = self.optimizer.get_optimized_config()

                        if optimization_result['changes_made']:
                            changes = ', '.join(optimization_result['changes_made'])
                            self.status_message.emit(f"Оптимизация: {changes}")  # type: ignore

                        # Отправляем отчет о производительности
                        performance_report = self.evaluator.get_performance_report()
                        self.performance_report.emit(performance_report)  # type: ignore

                        self.last_optimization = current_time
                    except Exception as e:
                        print(f"Ошибка оптимизации: {e}")

                time.sleep(0.5)

        except Exception as e:
            print(f"Критическая ошибка в основном цикле: {e}")
            self.status_message.emit(f"Критическая ошибка: {e}")  # type: ignore
        finally:
            loop.close()

    def _route_action(self, window_title: str, executable_name: str, classification: str):
        """Маршрутизирует действия на основе классификации"""
        display_name = f"{executable_name} ({window_title})" if executable_name else window_title
        display_name_short = display_name[:70]

        if classification == "ignored":
            self.status_message.emit(f"Активность: '{display_name_short}...' (Игнорируется)")  # type: ignore
            return

        # Проверяем рабочее время
        current_time = QTime.currentTime()
        start_time = QTime.fromString(self.current_settings["work_start_time"], "HH:mm")
        end_time = QTime.fromString(self.current_settings["work_end_time"], "HH:mm")

        if start_time.isValid() and end_time.isValid():
            if start_time <= end_time:
                is_work_time = start_time <= current_time <= end_time
            else:
                is_work_time = current_time >= start_time or current_time <= end_time
        else:
            is_work_time = True

        if not is_work_time and classification == "distraction":
            self.status_message.emit(f"Отвлечение ({display_name_short}...), но не рабочее время.")  # type: ignore
            return

        # Обрабатываем различные типы активности
        if classification == "distraction":
            target_info = f"{executable_name}|{window_title}"
            mode = self.current_settings.get("mode", "soft")

            if mode == "soft":
                message = (f"Обнаружено отвлечение: '{display_name_short}'. "
                           f"Время вернуться к работе!")
                self.action_command.emit("notify", message)  # type: ignore
                self.status_message.emit("Отвлечение: Уведомление отправлено.")  # type: ignore
            elif mode == "strict":
                # message = (f"Строгий режим: блокирую '{display_name_short}'")
                self.action_command.emit("block_or_close", target_info)  # type: ignore
                self.status_message.emit(f"Отвлечение: Блокировка активирована.")  # type: ignore

        elif classification == "productive":
            self.status_message.emit(f"Продуктивная активность: '{display_name_short}...'")  # type: ignore
        elif classification == "unknown":
            self.status_message.emit(f"Неизвестная активность: '{display_name_short}...'")  # type: ignore

    def _handle_screenshot_distraction(self, analysis: Dict):
        """Обрабатывает отвлечение, обнаруженное на скриншоте"""
        try:
            level = analysis.get('distraction_level', 'UNKNOWN')
            keywords = analysis.get('found_keywords', [])
            categories = analysis.get('categories', [])

            # Проверяем рабочее время
            current_time = QTime.currentTime()
            start_time = QTime.fromString(self.current_settings["work_start_time"], "HH:mm")
            end_time = QTime.fromString(self.current_settings["work_end_time"], "HH:mm")

            if start_time.isValid() and end_time.isValid():
                if start_time <= end_time:
                    is_work_time = start_time <= current_time <= end_time
                else:
                    is_work_time = current_time >= start_time or current_time <= end_time
            else:
                is_work_time = True

            if not is_work_time:
                self.status_message.emit(f"Отвлечение на экране ({level}), но не рабочее время.")  # type: ignore
                return

            # Формируем информацию об отвлечении
            distraction_info = f"Уровень: {level}"
            if categories:
                distraction_info += f", Категории: {', '.join(categories)}"
            if keywords:
                distraction_info += f", Найдено: {', '.join(keywords[:3])}"

            mode = self.current_settings.get("mode", "soft")

            if mode == "soft":
                message = f"Отвлечение на экране! {distraction_info}. Время сосредоточиться!"
                self.action_command.emit("notify", message)  # type: ignore
                self.status_message.emit(f"Скриншот-отвлечение ({level}): Уведомление.")  # type: ignore
            elif mode == "strict":
                # message = f"Строгий режим: обнаружено отвлечение на экране! {distraction_info}"
                self.action_command.emit("close_active_window", "")  # type: ignore
                self.status_message.emit(f"Скриншот-отвлечение ({level}): Принимаю меры.")  # type: ignore

        except Exception as e:
            print(f"Ошибка обработки скриншот-отвлечения: {e}")

    def get_system_statistics(self) -> Dict:
        """Возвращает статистику работы системы"""
        try:
            # Статистика производительности
            performance_report = self.evaluator.get_performance_report()

            # Статистика маршрутизатора
            router_stats = {}
            for task_type, stats in self.task_router.processing_stats.items():
                router_stats[task_type.value] = stats

            # Статистика скриншотов
            screenshot_stats = self._get_screenshot_statistics()

            # Текущая конфигурация
            current_config = self.optimized_config

            return {
                'performance': performance_report,
                'router_statistics': router_stats,
                'screenshot_analysis': screenshot_stats,
                'current_optimization': current_config,
                'system_status': {
                    'agents_active': len([a for a in self.task_router.agents.values() if not a.is_busy]),
                    'total_agents': len(self.task_router.agents),
                    'queue_size': len(self.window_data_queue),
                    'optimization_history': len(self.optimizer.optimization_history)
                }
            }
        except Exception as e:
            return {'error': f'Ошибка получения статистики: {e}'}

    def _get_screenshot_statistics(self) -> Dict:
        """Возвращает статистику анализа скриншотов"""
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
                [s.get('score', 0) for s in self.screenshot_history]) if self.screenshot_history else 0,
            'recent_trend': self._calculate_distraction_trend()
        }

    def _calculate_distraction_trend(self) -> str:
        """Вычисляет тренд отвлечений за последнее время"""
        if len(self.screenshot_history) < 10:
            return "Недостаточно данных"

        recent_10 = self.screenshot_history[-10:]
        previous_10 = self.screenshot_history[-20:-10] if len(self.screenshot_history) >= 20 else []

        if not previous_10:
            return "Недостаточно данных для тренда"

        recent_distraction_rate = sum(1 for s in recent_10 if s.get('is_distracted', False)) / len(recent_10)
        previous_distraction_rate = sum(1 for s in previous_10 if s.get('is_distracted', False)) / len(previous_10)

        if recent_distraction_rate > previous_distraction_rate + 0.1:
            return "Увеличение отвлечений"
        elif recent_distraction_rate < previous_distraction_rate - 0.1:
            return "Снижение отвлечений"
        else:
            return "Стабильный уровень"

    def update_settings(self):
        """Обновляет настройки системы"""
        self.current_settings = load_settings()

        # Обновляем агентов с новыми настройками
        if TaskType.KEYWORD_CLASSIFICATION in self.task_router.agents:
            keyword_agent = self.task_router.agents[TaskType.KEYWORD_CLASSIFICATION]
            keyword_agent.settings = self.current_settings

        # Обновляем оптимизированную конфигурацию
        self.optimized_config = self.optimizer.get_optimized_config()

        self.status_message.emit("Настройки обновлены с оптимизацией")  # type: ignore
        print("AttentionAnalyzer: Настройки обновлены")

    def stop(self):
        """Останавливает анализатор"""
        self._running = False

        # Завершаем executor
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

        self.wait()
        self.status_message.emit("Улучшенный анализатор внимания остановлен")  # type: ignore

        # Финальный отчет
        try:
            final_stats = self.get_system_statistics()
            print("=== ФИНАЛЬНАЯ СТАТИСТИКА СИСТЕМЫ ===")
            print(json.dumps(final_stats, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Ошибка получения финальной статистики: {e}")

