"""Download the final pinned MAEST model for Windows source-code users."""

from pathlib import Path

from desktop_runtime import prepare_desktop_model


def console_progress():
    state = {"message": None, "percent": -10}

    def report(message, fraction):
        if fraction is None:
            if message != state["message"]:
                print(message, flush=True)
                state["message"] = message
            return
        percent = int(fraction * 100)
        if message != state["message"] or percent >= state["percent"] + 10 or percent == 100:
            print(f"{message}: {percent}%", flush=True)
            state.update(message=message, percent=percent)

    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prepare_desktop_model(root, callback=console_progress())
