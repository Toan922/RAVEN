"""
Entry point for the RAVEN operator console.

    python -m gui            normal mode: real Redis server and the YOLO model
    python -m gui --demo     demo mode: built-in fake Redis, simulated detector, generated test flight
"""
import argparse
import sys
from pathlib import Path

# make the project root importable so backend/ and discord_bot/ can be used from the GUI
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(prog="python -m gui", description="RAVEN operator console")
    parser.add_argument("--demo", action="store_true",
                        help="run without Redis or the model: fake Redis, simulated detector, generated video")
    args, qt_args = parser.parse_known_args()

    from dotenv import load_dotenv
    from PySide6.QtWidgets import QApplication

    # QtWebEngine has to be imported before the QApplication is created
    from gui.main_window import MainWindow

    # read DISCORD_WEBHOOK_URL (and anything else) from the project's .env
    load_dotenv(ROOT_DIR / ".env")

    app = QApplication([sys.argv[0], *qt_args])
    app.setOrganizationName("RAVEN")
    # demo mode keeps its own saved settings so it never overwrites the real ones
    app.setApplicationName("RAVEN Operator Console" + (" Demo" if args.demo else ""))

    demo = None
    if args.demo:
        from gui.demo import DemoBackend

        print("Starting demo mode (generating the demo flight the first time)…", flush=True)
        demo = DemoBackend()
        demo.start()
        app.aboutToQuit.connect(demo.stop)

    window = MainWindow(demo=demo)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
