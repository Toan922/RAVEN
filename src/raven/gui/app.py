"""
Entry point for the RAVEN operator console.

    python -m raven.gui            normal mode: the YOLO model on the drone feeds set in Settings
    python -m raven.gui --demo     demo mode: simulated detector, generated test flights

Needs RAVEN installed (pip install -e . from the repo root) so the raven package can be imported.

The console encrypts its data folder (raven.backend.vault) and asks for a sign-in before it opens; the first
launch creates the administrator account.
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

    from PySide6.QtWidgets import QDialog, QMessageBox

    from raven.backend import vault
    from raven.backend.accounts import UserStore
    from raven.gui.sign_in import SignInDialog
    from raven.gui.theme import apply_theme, system_theme

    apply_theme(app, system_theme())

    # saved frames, the detection log, the Discord outbox and the accounts are encrypted with a key kept in the
    # Windows credential store
    try:
        vault.enable()
        accounts = UserStore()
    except Exception as error:  # no credential store, or a data folder encrypted with another key
        QMessageBox.critical(None, "RAVEN can't open its data",
                             f"The console couldn't unlock its encrypted data folder:\n\n{error}")
        sys.exit(1)

    sign_in = SignInDialog(accounts, demo=args.demo)
    if sign_in.exec() != QDialog.Accepted or sign_in.user is None:
        sys.exit(0)

    demo = None
    if args.demo:
        from raven.gui.demo import DemoBackend

        print("Starting demo mode (generating the demo flight the first time)…", flush=True)
        demo = DemoBackend()
        demo.start()

    window = MainWindow(demo=demo, user=sign_in.user, accounts=accounts)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
