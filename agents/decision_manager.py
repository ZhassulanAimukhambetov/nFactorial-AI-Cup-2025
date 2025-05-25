import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
from PyQt5.QtCore import QTime, pyqtSignal, QObject
import logging


class ActionType(Enum):
    """Типы действий системы"""
    NOTIFY = "notify"
    BLOCK_APP = "block_app"
    CLOSE_WINDOW = "close_window"
    LOG_ACTIVITY = "log_activity"
    NO_ACTION = "no_action"


class ViolationSeverity(Enum):
    """Уровень серьезности нарушения"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Decision:
    """Решение системы"""
    action_type: ActionType
    severity: ViolationSeverity
    target: str = ""
    message: str = ""
    reason: str = ""
    confidence: float = 1.0
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class DecisionManager(QObject):
    """Менеджер принятия решений для системы контроля внимания"""

    # Сигналы для взаимодействия с основным приложением
    action_required = pyqtSignal(str, str)  # action_type, data
    notification_request = pyqtSignal(str)  # message
    status_update = pyqtSignal(str)  # status message

    def __init__(self, settings: Dict[str, Any]):
        super().__init__()
        self.settings = settings
        self.decision_history = []
        self.violation_counts = {}
        self.last_notification_time = {}

        # Настройки уведомлений
        self.notification_cooldown = 30  # секунд между уведомлениями для одного типа
        self.max_violations_per_hour = 10

        # Счетчики для статистики
        self.stats = {
            'total_decisions': 0,
            'notifications_sent': 0,
            'apps_blocked': 0,
            'windows_closed': 0
        }

    def make_decision(self, analysis_data: Dict[str, Any]) -> Decision:
        """Принимает решение на основе результатов анализа"""
        try:
            classification = analysis_data.get('classification', 'unknown')
            window_title = analysis_data.get('window_title', '')
            executable_name = analysis_data.get('executable_name', '')
            confidence = analysis_data.get('confidence', 1.0)

            # Проверяем рабочее время
            if not self._is_work_time():
                return self._create_decision(
                    ActionType.NO_ACTION,
                    ViolationSeverity.LOW,
                    reason="Не рабочее время"
                )

            if self._should_ignore_activity(window_title, executable_name):
                return self._handle_ignored_activity(window_title, executable_name)

            elif classification == "ignored":
                return self._handle_ignored_activity(window_title, executable_name)

            elif classification == "productive":
                return self._handle_productive_activity(window_title, executable_name)

            elif classification == "distraction":
                return self._handle_distraction(window_title, executable_name, confidence)

            else:  # unknown
                return self._handle_unknown_activity(window_title, executable_name)

        except Exception as e:
            logging.error(f"Ошибка принятия решения: {e}")
            return self._create_decision(
                ActionType.NO_ACTION,
                ViolationSeverity.LOW,
                reason=f"Ошибка анализа: {str(e)}"
            )

    def make_screenshot_decision(self, screenshot_analysis: Dict[str, Any]) -> Decision:
        """Принимает решение на основе анализа скриншота"""
        try:
            is_distracted = screenshot_analysis.get('is_distracted', False)
            distraction_level = screenshot_analysis.get('distraction_level', 'LOW')
            categories = screenshot_analysis.get('categories', [])

            if not self._is_work_time():
                return self._create_decision(
                    ActionType.NO_ACTION,
                    ViolationSeverity.LOW,
                    reason="Не рабочее время"
                )

            if not is_distracted:
                return self._create_decision(
                    ActionType.LOG_ACTIVITY,
                    ViolationSeverity.LOW,
                    reason="Отвлечения не обнаружены"
                )

            # Определяем серьезность на основе уровня отвлечения
            severity_map = {
                'LOW': ViolationSeverity.LOW,
                'MEDIUM': ViolationSeverity.MEDIUM,
                'HIGH': ViolationSeverity.HIGH
            }
            severity = severity_map.get(distraction_level, ViolationSeverity.MEDIUM)

            # Формируем сообщение
            categories_str = ', '.join(categories) if categories else 'общие'
            message = f"Обнаружено отвлечение на экране! Уровень: {distraction_level}, Категории: {categories_str}"

            mode = self.settings.get("mode", "soft")

            if mode == "soft":
                return self._create_decision(
                    ActionType.NOTIFY,
                    severity,
                    message=message,
                    reason=f"Скриншот-анализ: {distraction_level}"
                )
            elif mode == "strict":
                return self._create_decision(
                    ActionType.CLOSE_WINDOW,
                    severity,
                    target="active_window",
                    message=f"Строгий режим: {message}",
                    reason=f"Скриншот-анализ: {distraction_level}"
                )

            return self._create_decision(ActionType.NO_ACTION, ViolationSeverity.LOW)

        except Exception as e:
            logging.error(f"Ошибка анализа скриншота: {e}")
            return self._create_decision(
                ActionType.NO_ACTION,
                ViolationSeverity.LOW,
                reason=f"Ошибка анализа скриншота: {str(e)}"
            )

    def _handle_distraction(self, window_title: str, executable_name: str, confidence: float) -> Decision:
        """Обрабатывает отвлекающую активность"""
        target_info = f"{executable_name}|{window_title}"
        display_name = self._get_display_name(window_title, executable_name)

        if self._should_ignore_activity(window_title, executable_name):
            return self._handle_ignored_activity(window_title, executable_name)

        # Определяем серьезность на основе confidence и истории нарушений
        violation_key = executable_name or window_title
        violation_count = self.violation_counts.get(violation_key, 0)

        if violation_count >= 3:
            severity = ViolationSeverity.HIGH
        elif confidence > 0.9:
            severity = ViolationSeverity.MEDIUM
        else:
            severity = ViolationSeverity.LOW

        # Увеличиваем счетчик нарушений
        self.violation_counts[violation_key] = violation_count + 1

        mode = self.settings.get("mode", "soft")

        if mode == "soft":
            message = f"Обнаружено отвлечение: '{display_name}'. Время вернуться к работе!"
            return self._create_decision(
                ActionType.NOTIFY,
                severity,
                target=target_info,
                message=message,
                reason="Отвлекающая активность",
                confidence=confidence
            )

        elif mode == "strict":
            message = f"Строгий режим: блокирую '{display_name}'"
            return self._create_decision(
                ActionType.BLOCK_APP,
                severity,
                target=target_info,
                message=message,
                reason="Отвлекающая активность (строгий режим)",
                confidence=confidence
            )

        return self._create_decision(ActionType.NO_ACTION, ViolationSeverity.LOW)

    def _handle_productive_activity(self, window_title: str, executable_name: str) -> Decision:
        """Обрабатывает продуктивную активность"""
        display_name = self._get_display_name(window_title, executable_name)

        return self._create_decision(
            ActionType.LOG_ACTIVITY,
            ViolationSeverity.LOW,
            reason=f"Продуктивная активность: {display_name}"
        )

    def _handle_ignored_activity(self, window_title: str, executable_name: str) -> Decision:
        """Обрабатывает игнорируемую активность"""
        display_name = self._get_display_name(window_title, executable_name)

        return self._create_decision(
            ActionType.NO_ACTION,
            ViolationSeverity.LOW,
            reason=f"Игнорируемая активность: {display_name}"
        )

    def _handle_unknown_activity(self, window_title: str, executable_name: str) -> Decision:
        """Обрабатывает неизвестную активность"""
        display_name = self._get_display_name(window_title, executable_name)

        if self._should_ignore_activity(window_title, executable_name):
            return self._handle_ignored_activity(window_title, executable_name)

        return self._create_decision(
            ActionType.LOG_ACTIVITY,
            ViolationSeverity.LOW,
            reason=f"Неизвестная активность: {display_name}"
        )

    def execute_decision(self, decision: Decision):
        """Выполняет принятое решение"""
        try:
            self.decision_history.append(decision)
            self.stats['total_decisions'] += 1

            if decision.action_type == ActionType.NOTIFY:
                if self._should_send_notification("distraction"):
                    self.notification_request.emit(decision.message)  # type: ignore
                    self.stats['notifications_sent'] += 1
                    self.status_update.emit(f"Уведомление: {decision.reason}")  # type: ignore

            elif decision.action_type == ActionType.BLOCK_APP:
                self.action_required.emit("block_or_close", decision.target)  # type: ignore
                self.stats['apps_blocked'] += 1
                self.status_update.emit(f"Блокировка: {decision.reason}")  # type: ignore

            elif decision.action_type == ActionType.CLOSE_WINDOW:
                self.action_required.emit("close_active_window", decision.target)  # type: ignore
                self.stats['windows_closed'] += 1
                self.status_update.emit(f"Закрытие окна: {decision.reason}")  # type: ignore

            elif decision.action_type == ActionType.LOG_ACTIVITY:
                self.status_update.emit(decision.reason)  # type: ignore

            elif decision.action_type == ActionType.NO_ACTION:
                # Ничего не делаем, но логируем
                if decision.reason:
                    self.status_update.emit(decision.reason)  # type: ignore

        except Exception as e:
            logging.error(f"Ошибка выполнения решения: {e}")
            self.status_update.emit(f"Ошибка выполнения действия: {str(e)}")  # type: ignore

    def _is_work_time(self) -> bool:
        """Проверяет, является ли текущее время рабочим"""
        try:
            current_time = QTime.currentTime()
            start_time = QTime.fromString(self.settings.get("work_start_time", "09:00"), "HH:mm")
            end_time = QTime.fromString(self.settings.get("work_end_time", "18:00"), "HH:mm")

            if not start_time.isValid() or not end_time.isValid():
                return True  # Если время не задано, считаем что всегда рабочее

            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:
                # Переход через полночь
                return current_time >= start_time or current_time <= end_time

        except Exception:
            return True

    def _should_send_notification(self, notification_type: str) -> bool:
        """Проверяет, следует ли отправлять уведомление (учитывает кулдаун)"""
        current_time = time.time()
        last_time = self.last_notification_time.get(notification_type, 0)

        if current_time - last_time >= self.notification_cooldown:
            self.last_notification_time[notification_type] = current_time
            return True

        return False

    def _get_display_name(self, window_title: str, executable_name: str) -> str:
        """Формирует отображаемое имя для активности"""
        if executable_name and window_title:
            return f"{executable_name} ({window_title[:50]}{'...' if len(window_title) > 50 else ''})"
        elif executable_name:
            return executable_name
        elif window_title:
            return window_title[:50] + ('...' if len(window_title) > 50 else '')
        else:
            return "Неизвестное приложение"

    def _create_decision(self, action_type: ActionType, severity: ViolationSeverity,
                         target: str = "", message: str = "", reason: str = "",
                         confidence: float = 1.0) -> Decision:
        """Создает объект решения"""
        return Decision(
            action_type=action_type,
            severity=severity,
            target=target,
            message=message,
            reason=reason,
            confidence=confidence
        )

    def update_settings(self, new_settings: Dict[str, Any]):
        """Обновляет настройки менеджера"""
        self.settings = new_settings
        self.status_update.emit("Настройки DecisionManager обновлены")  # type: ignore

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает полную статистику работы менеджера решений"""
        # Базовые статистические данные
        stats = {
            'total_decisions': len(self.decision_history),
            'performance_stats': {
                'notifications_sent': self.stats.get('notifications_sent', 0),
                'apps_blocked': self.stats.get('apps_blocked', 0),
                'windows_closed': self.stats.get('windows_closed', 0),
                'errors_occurred': self.stats.get('errors_occurred', 0)
            },
            'violation_counts': dict(self.violation_counts),
            'current_mode': self.settings.get('mode', 'soft'),
            'system_status': {
                'is_work_time': self._is_work_time(),
                'last_decision_time': self.decision_history[-1].timestamp if self.decision_history else None
            }
        }

        # Если нет истории решений, возвращаем базовую статистику
        if not self.decision_history:
            stats['message'] = 'Решения еще не принимались'
            return stats

        # Статистика по типам действий
        action_stats = {}
        for action in ActionType:
            action_stats[action.value] = sum(
                1 for d in self.decision_history
                if d.action_type == action
            )

        # Статистика по уровням серьезности
        severity_stats = {}
        for severity in ViolationSeverity:
            severity_stats[severity.name.lower()] = sum(
                1 for d in self.decision_history
                if d.severity == severity
            )

        # Последние 5 решений
        recent_decisions = []
        for decision in self.decision_history[-5:]:
            recent_decisions.append({
                'action': decision.action_type.value,
                'severity': decision.severity.name,
                'target': decision.target[:50] + ('...' if len(decision.target) > 50 else ''),
                'reason': decision.reason[:100] + ('...' if len(decision.reason) > 100 else ''),
                'timestamp': decision.timestamp,
                'confidence': round(decision.confidence, 2)
            })

        # Самые частые нарушения
        top_violations = sorted(
            self.violation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # Объединяем все данные
        stats.update({
            'action_statistics': action_stats,
            'severity_statistics': severity_stats,
            'recent_decisions': recent_decisions,
            'top_violations': [
                {'target': k, 'count': v}
                for k, v in top_violations
            ],
            'decision_trend': self._get_decision_trend()
        })

        return stats

    def _get_decision_trend(self) -> str:
        """Определяет тренд принятия решений"""
        if len(self.decision_history) < 10:
            return "Недостаточно данных"

        # Анализируем последние 10 решений
        recent = self.decision_history[-10:]
        blocks = sum(1 for d in recent if d.action_type == ActionType.BLOCK_APP)
        notifies = sum(1 for d in recent if d.action_type == ActionType.NOTIFY)

        if blocks >= 3:
            return "Высокая активность блокировок"
        elif notifies >= 5:
            return "Много уведомлений"
        elif not any(d.action_type in (ActionType.BLOCK_APP, ActionType.NOTIFY) for d in recent):
            return "Низкая активность"
        else:
            return "Стабильная работа"

    def reset_statistics(self):
        """Сбрасывает статистику"""
        self.decision_history.clear()
        self.violation_counts.clear()
        self.last_notification_time.clear()
        self.stats = {
            'total_decisions': 0,
            'notifications_sent': 0,
            'apps_blocked': 0,
            'windows_closed': 0
        }
        self.status_update.emit("Статистика DecisionManager сброшена")  # type: ignore

    def _should_ignore_activity(self, window_title: str, executable_name: str) -> bool:
        """
        НОВЫЙ МЕТОД: Проверяет, следует ли игнорировать данную активность
        на основе настроек игнорируемых ключевых слов
        """
        ignored_keywords = self.settings.get('ignored_keywords', [])

        if not ignored_keywords:
            return False

        # Приводим к нижнему регистру для сравнения
        window_title_lower = window_title.lower() if window_title else ""
        executable_name_lower = executable_name.lower() if executable_name else ""

        # Проверяем каждое игнорируемое ключевое слово
        for keyword in ignored_keywords:
            keyword_lower = keyword.lower()

            # Проверяем наличие ключевого слова в заголовке окна или имени исполняемого файла
            if (keyword_lower in window_title_lower or
                    keyword_lower in executable_name_lower):
                logging.info(f"Активность игнорируется по ключевому слову '{keyword}': "
                             f"окно='{window_title}', exe='{executable_name}'")
                return True

        return False

