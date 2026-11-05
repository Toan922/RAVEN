"""
Entry point for the RAVEN operator console.

    python -m raven.gui            normal mode: the YOLO model on the drone feeds set in Settings
    python -m raven.gui --demo     demo mode: simulated detector, generated test flights

Needs RAVEN installed (pip install -e . from the repo root) so the raven package can be imported.
"""
import argparse
import sys

from raven.backend.config import ROOT_DIR


def main():
    parser = argparse.ArgumentParser(prog="python -m raven.gui", description="RAVEN operator console")
    parser.add_argument("--demo", action="store_true",
                        help="run without the model: simulated detector, generated test flights")
    args, qt_args = parser.parse_known_args()

    from dotenv import load_dotenv
    from PySide6.QtWidgets import QApplication

    # QtWebEngine has to be imported before the QApplication is created
    from raven.gui.main_window import MainWindow

    # read DISCORD_WEBHOOK_URL (and anything else) from the project's .env
    load_dotenv(ROOT_DIR / ".env")

    app = QApplication([sys.argv[0], *qt_args])
    app.setOrganizationName("RAVEN")
    # demo mode keeps its own saved settings so it never overwrites the real ones
    app.setApplicationName("RAVEN Operator Console" + (" Demo" if args.demo else ""))

    demo = None
    if args.demo:
        from raven.gui.demo import DemoBackend

        print("Starting demo mode (generating the demo flight the first time)…", flush=True)
        demo = DemoBackend()
        demo.start()

    window = MainWindow(demo=demo)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
