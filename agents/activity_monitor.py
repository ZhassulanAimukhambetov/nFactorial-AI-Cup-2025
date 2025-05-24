import time
import pygetwindow as gw
import psutil
from PyQt5.QtCore import QThread, pyqtSignal
import ctypes


# Предполагается, что код выполняется в Windows.


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

                    try:
                        if hasattr(active_window, '_hWnd'):
                            hwnd = active_window._hWnd
                            if hwnd:
                                dw_process_id = ctypes.c_ulong()
                                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(dw_process_id))
                                pid = dw_process_id.value
                    except Exception as e_pid:
                        pid = None  # PID не удалось получить
                    if pid:
                        try:
                            process = psutil.Process(pid)
                            current_executable_name = process.name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            current_executable_name = ""
                    else:
                        current_executable_name = ""

                if current_window_title != last_window_data["title"] or \
                        current_executable_name != last_window_data["executable"]:
                    self.active_window_changed.emit(current_window_title, current_executable_name)
                    last_window_data["title"] = current_window_title
                    last_window_data["executable"] = current_executable_name

            except gw.PyGetWindowException as e:
                self.status_message.emit("Ошибка мониторинга окна. Проверьте разрешения.")
            except Exception as e:
                print(f"Неизвестная ошибка в ActivityMonitor: {e}")
                self.status_message.emit("Произошла ошибка в мониторе активности.")

            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self.wait()
        self.status_message.emit("Мониторинг активности остановлен.")
