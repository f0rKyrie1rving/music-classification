"""Exercise actual Tk buttons with the included audio; optional hold for visual QA."""

import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import desktop_app


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--close-file", type=Path, help="Hold the verified UI until this file exists.")
    args = parser.parse_args()
    if args.close_file and args.close_file.exists():
        raise ValueError("Close marker already exists; choose a new marker.")
    root = tk.Tk()
    app = desktop_app.MusicTaggingApp(root)
    closing = []
    root.protocol("WM_DELETE_WINDOW", lambda: closing.append(True))
    record = {"state": "started", "platform": sys.platform, "native_tk": True}
    try:
        with patch.object(desktop_app, "run_prediction", wraps=desktop_app.run_prediction) as predict:
            root.update()
            app.sample_button.invoke()
            assert Path(app.audio_path.get()) == desktop_app.SAMPLE_AUDIO
            app.analyze_button.invoke()
            deadline = time.monotonic() + 120
            while app.last_result is None and time.monotonic() < deadline:
                root.update()
                time.sleep(0.05)
            assert app.last_result is not None, app.status.get()
            assert app.status.get() == "Analysis complete"
            assert root.title() == "Music Tagging Demo 1.2.0"
            assert app.selected_tags.get() == "v1.2.0 · Selected tags: pop, rock"
            assert tuple(app.results.item("pop", "values")) == ("0.3219", "0.2750", "Yes")
            assert tuple(app.results.item("ambient", "values")) == ("0.0272", "0.2000", "No")
            current = {label: list(app.results.item(label, "values"))
                       for label in ("electronic", "pop", "ambient", "rock")}
            app.compare_checkbox.invoke()
            root.update()
            assert app.results.heading("score", "text") == "Estimate (v1.1.0)"
            assert app.selected_tags.get() == "v1.1.0 · Selected tags: rock"
            assert tuple(app.results.item("pop", "values")) == ("0.2483", "0.2750", "No")
            assert tuple(app.results.item("ambient", "values")) == ("0.0078", "0.1321", "No")
            old = {label: list(app.results.item(label, "values")) for label in current}
            app.compare_checkbox.invoke()
            root.update()
            assert app.results.heading("score", "text") == "Estimate (v1.2.0)"
            assert app.selected_tags.get() == "v1.2.0 · Selected tags: pop, rock"
            assert {label: list(app.results.item(label, "values")) for label in current} == current
            assert predict.call_count == 1
            record.update(state="passed", version=desktop_app.APP_VERSION,
                          inference_calls=predict.call_count, current_rows=current,
                          baseline_rows=old, actual_audio=str(desktop_app.SAMPLE_AUDIO),
                          checks="Actual sample/analyze/toggle buttons; title, all rows, selected tags, and toggle-back without new inference")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
        if args.close_file:
            while not closing and not args.close_file.exists():
                root.update()
                time.sleep(0.1)
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
