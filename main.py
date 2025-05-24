import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


if __name__ == "__main__":
    # Создаем экземпляр QApplication
    # QApplication управляет ресурсами приложения и циклом событий.
    app = QApplication(sys.argv)

    # Создаем экземпляр нашего главного окна
    main_win = MainWindow()

    # Показываем главное окно
    main_win.show()

    # Запускаем цикл событий приложения.
    # sys.exit() гарантирует, что приложение завершится корректно,
    # когда главное окно будет закрыто.
    sys.exit(app.exec_())
