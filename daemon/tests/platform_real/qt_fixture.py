"""A Qt window with a text field and a password field (PySide6), for apps_fields_check.py.

Prints ``ready`` once shown with the text field focused; quits when killed.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget


def main() -> None:
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Heliox field fixture")
    layout = QVBoxLayout(window)
    message = QLineEdit()
    message.setAccessibleName("message")
    password = QLineEdit()
    password.setAccessibleName("password")
    password.setEchoMode(QLineEdit.EchoMode.Password)
    layout.addWidget(message)
    layout.addWidget(password)
    window.resize(480, 160)
    window.show()
    window.activateWindow()
    message.setFocus()

    def ready() -> None:
        message.setFocus()
        print("ready", flush=True)

    QTimer.singleShot(600, ready)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
