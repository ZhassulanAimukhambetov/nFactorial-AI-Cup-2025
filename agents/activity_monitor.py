import time
import pygetwindow as gw
import psutil
from PyQt5.QtCore import QThread, pyqtSignal

# Предполагается, что код выполняется в Windows.
# Для кроссплатформенности может потребоваться другая логика.
import ctypes

class ActivityMonitor(QThread):
    active_window_changed = pyqtSignal(str, str)
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.interval = 1

    def run(self):
        self.status_message.emit("Мониторинг активности запущен...")
        last_window_data = {"title": "", "executable": ""}

        while self._running:
            try:
                active_window = gw.getActiveWindow()
                current_window_title = ""
                current_executable_name = ""
                pid = None

                if active_window:
                    current_window_title = active_window.title

                    # --- НАЧАЛО ИЗМЕНЕНИЯ: Получение PID через HWND ---
                    try:
                        # Проверяем, есть ли атрибут _hWnd (специфично для Windows)
                        if hasattr(active_window, '_hWnd'):
                            hwnd = active_window._hWnd
                            if hwnd: # Убедимся, что хэндл не нулевой
                                dw_process_id = ctypes.c_ulong()
                                # Вызов функции Windows API GetWindowThreadProcessId
                                # https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getwindowthreadprocessid
                                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(dw_process_id))
                                pid = dw_process_id.value
                        # Если нужно добавить поддержку других ОС, здесь будут другие ветки
                    except Exception as e_pid:
                        # print(f"Ошибка при получении PID для окна '{current_window_title}': {e_pid}")
                        pid = None # PID не удалось получить
                    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

                    if pid:
                        try:
                            process = psutil.Process(pid)
                            current_executable_name = process.name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            # Процесс мог завершиться, или нет доступа
                            current_executable_name = ""
                            # print(f"Не удалось получить имя процесса для PID: {pid}")
                    else:
                        # PID не был получен (например, окно без HWND или ошибка API)
                        current_executable_name = ""
                        # print(f"PID для окна '{current_window_title}' не определен.")


                if current_window_title != last_window_data["title"] or \
                        current_executable_name != last_window_data["executable"]:
                    self.active_window_changed.emit(current_window_title, current_executable_name)
                    last_window_data["title"] = current_window_title
                    last_window_data["executable"] = current_executable_name

            except gw.PyGetWindowException as e:
                # print(f"Ошибка pygetwindow: {e}")
                self.status_message.emit("Ошибка мониторинга окна. Проверьте разрешения.")
            except Exception as e:
                print(f"Неизвестная ошибка в ActivityMonitor: {e}") # Эта строка выводила вашу ошибку
                self.status_message.emit("Произошла ошибка в мониторе активности.")

            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self.wait()
        self.status_message.emit("Мониторинг активности остановлен.")