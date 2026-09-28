"""Small desktop interface for the final four-label music tagging model."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path

from application_release import APP_VERSION
from desktop_runtime import configure_runtime, model_files_present, prepare_desktop_model


BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# Source checkouts keep the established repository-local model directory.
# Frozen installers use a durable per-user directory instead of PyInstaller's
# temporary/read-only bundle location.
APP_DATA_ROOT, MODEL_DIR = configure_runtime(
    None if getattr(sys, "frozen", False) else BUNDLE_ROOT
)
SAMPLE_AUDIO = BUNDLE_ROOT / "data/previews/track_0207501_30s.wav"


def score_display_rows(result, compare_v1_1=False):
    """Format one cached model's scores, thresholds, and original decisions."""
    displayed = result["baseline"] if compare_v1_1 else result
    rows = []
    for item in displayed["scores"]:
        score, threshold = item["score"], item["threshold"]
        rows.append(
            (item["label"], (f"{score:.4f}", f"{threshold:.4f}", "Yes" if item["selected"] else "No"))
        )
    return rows


def run_prediction(audio, callback=None):
    """Prepare the model if needed, then classify one audio file."""
    # Import only after configure_runtime has redirected model and cache paths.
    from application_predict import classify, load_application

    # Reject missing or altered release files before a potentially large download.
    load_application(root=BUNDLE_ROOT)
    prepare_desktop_model(data_root=APP_DATA_ROOT, callback=callback)
    from maest_hf_features import HfMaestEncoder

    if callback is not None:
        callback("Loading MAEST encoder", None)
    encoder = HfMaestEncoder(device="cpu")
    if callback is not None:
        callback("Analyzing the first 30 seconds", None)
    return classify(
        audio, device="cpu", encoder=encoder, compare_baseline=True, root=BUNDLE_ROOT
    )


class MusicTaggingApp:
    """Tk desktop UI; model work stays off the UI thread."""

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.events = queue.Queue()
        self.audio_path = tk.StringVar()
        self.last_result = None
        self.compare_v1_1 = tk.BooleanVar(value=False)
        self.score_note = tk.StringVar(
            value="Estimates suggest how likely each tag is; they can be wrong."
        )
        self.status = tk.StringVar(
            value="Model ready" if model_files_present(MODEL_DIR) else "Model downloads on first analysis"
        )
        self.selected_tags = tk.StringVar(value="Select an audio file to begin.")

        root.title(f"Music Tagging Demo {APP_VERSION}")
        root.geometry("820x650")
        root.minsize(720, 580)
        root.option_add("*Font", ("Segoe UI", 10))

        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22))
        # Inherit the native foreground so status/help text stays readable in dark mode.
        style.configure("Result.TLabel", font=("Segoe UI Semibold", 13))

        outer = ttk.Frame(root, padding=24)
        outer.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)

        ttk.Label(outer, text="Multi-label Music Tagging", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text=(
                "Academic portfolio demo by Chenglin Song · electronic, pop, ambient, rock · "
                "first 30 seconds"
            ),
            style="Subtitle.TLabel",
            wraplength=760,
        ).grid(row=1, column=0, sticky="w", pady=(4, 20))

        chooser = ttk.Frame(outer)
        chooser.grid(row=2, column=0, sticky="ew")
        chooser.columnconfigure(0, weight=1)
        self.path_entry = ttk.Entry(chooser, textvariable=self.audio_path, state="readonly")
        self.path_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.browse_button = ttk.Button(chooser, text="Choose audio…", command=self.choose_audio)
        self.browse_button.grid(row=0, column=1, padx=(0, 8))
        self.sample_button = ttk.Button(chooser, text="Use included example", command=self.use_sample)
        self.sample_button.grid(row=0, column=2)

        actions = ttk.Frame(outer)
        actions.grid(row=3, column=0, sticky="ew", pady=(14, 10))
        actions.columnconfigure(1, weight=1)
        self.analyze_button = ttk.Button(actions, text="Analyze audio", command=self.analyze)
        self.analyze_button.grid(row=0, column=0, padx=(0, 14))
        self.progress = ttk.Progressbar(actions, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=1, sticky="ew")

        ttk.Label(outer, textvariable=self.selected_tags, style="Result.TLabel").grid(
            row=4, column=0, sticky="w", pady=(8, 8)
        )

        columns = ("score", "threshold", "decision")
        self.results = ttk.Treeview(outer, columns=columns, show="tree headings", height=6)
        self.results.heading("#0", text="Label")
        self.results.heading("score", text=f"Estimate (v{APP_VERSION})")
        self.results.heading("threshold", text="Threshold")
        self.results.heading("decision", text="Selected")
        self.results.column("#0", width=220, anchor="w")
        self.results.column("score", width=130, anchor="center")
        self.results.column("threshold", width=130, anchor="center")
        self.results.column("decision", width=130, anchor="center")
        self.results.grid(row=5, column=0, sticky="nsew")
        for label in ("electronic", "pop", "ambient", "rock"):
            self.results.insert("", "end", iid=label, text=label.title(), values=("—", "—", "—"))

        footer = ttk.Frame(outer)
        footer.grid(row=6, column=0, sticky="ew", pady=(14, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status, style="Subtitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            footer,
            textvariable=self.score_note,
            style="Subtitle.TLabel",
            wraplength=640,
        ).grid(row=2, column=0, sticky="w", pady=(3, 0))
        self.compare_checkbox = ttk.Checkbutton(
            footer,
            text="Compare v1.1",
            variable=self.compare_v1_1,
            command=self.refresh_result_view,
        )
        self.compare_checkbox.grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Button(footer, text="About", command=self.show_about).grid(
            row=0, column=1, rowspan=3, sticky="e"
        )

        root.after(100, self.process_events)

    def choose_audio(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[
                ("Supported audio", "*.wav *.flac *.ogg *.mp3"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.audio_path.set(path)
            self.status.set("Ready to analyze")

    def use_sample(self):
        if SAMPLE_AUDIO.is_file():
            self.audio_path.set(str(SAMPLE_AUDIO))
            self.status.set("Included CC BY 3.0 example selected")
        else:
            from tkinter import messagebox

            messagebox.showerror("Example unavailable", "The included example audio was not found.")

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.browse_button.configure(state=state)
        self.sample_button.configure(state=state)
        self.analyze_button.configure(state=state)

    def analyze(self):
        from tkinter import messagebox

        audio = Path(self.audio_path.get())
        if not audio.is_file():
            messagebox.showinfo("Choose audio", "Choose a WAV, FLAC, OGG, or MP3 file first.")
            return
        self.set_busy(True)
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)
        self.status.set("Preparing analysis…")
        threading.Thread(target=self._prediction_worker, args=(audio,), daemon=True).start()

    def _prediction_worker(self, audio):
        def report(message, fraction=None):
            self.events.put(("progress", (message, fraction)))

        try:
            result = run_prediction(audio, report)
        except Exception as error:  # UI boundary: surface a readable failure dialog.
            self.events.put(("error", str(error)))
        else:
            self.events.put(("result", result))

    def process_events(self):
        from tkinter import messagebox

        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    message, fraction = payload
                    self.status.set(message)
                    if fraction is not None:
                        self.progress.stop()
                        self.progress.configure(mode="determinate", value=100 * fraction)
                elif kind == "result":
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=100)
                    self.last_result = payload
                    self.refresh_result_view()
                    self.status.set("Analysis complete")
                    self.set_busy(False)
                elif kind == "error":
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    self.status.set("Analysis failed")
                    self.set_busy(False)
                    messagebox.showerror("Could not analyze audio", payload)
        except queue.Empty:
            pass
        self.root.after(100, self.process_events)

    def refresh_result_view(self):
        """Switch the displayed values from the cached prediction only."""
        compare = self.compare_v1_1.get()
        displayed = None
        if self.last_result is not None:
            displayed = self.last_result["baseline"] if compare else self.last_result
        version = displayed["application_version"] if displayed is not None else (
            "1.1.0" if compare else APP_VERSION
        )
        self.results.heading("score", text=f"Estimate (v{version})")
        self.score_note.set(
            "Showing v1.1 estimates, thresholds, and selected tags for the same audio."
            if compare else (
                f"Showing v{APP_VERSION}. Estimates suggest how likely each tag is; "
                "they can be wrong."
            )
        )
        if displayed is not None:
            tags = displayed["predicted_tags"]
            self.selected_tags.set(
                f"v{version} · Selected tags: " + (", ".join(tags) if tags else "none")
            )
            for label, values in score_display_rows(self.last_result, compare):
                self.results.item(label, values=values)

    def show_about(self):
        from tkinter import messagebox

        messagebox.showinfo(
            "About Music Tagging Demo",
            (
                f"Version {APP_VERSION}\n\n"
                "Discogs-MAEST block 7 with four locally trained logistic-regression heads.\n"
                "Academic, research, and portfolio use; not a production classifier.\n\n"
                "Author: Chenglin Song (宋承麟)\n"
                "Project: github.com/f0rKyrie1rving/music-classification"
            ),
        )


def smoke_test(audio, output):
    """Run the packaged Windows executable end-to-end in release CI."""
    output = Path(output)
    try:
        result = run_prediction(Path(audio))
        payload = {"ok": True, "result": result}
        code = 0
    except Exception as error:
        payload = {"ok": False, "error": str(error)}
        code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", type=Path, metavar="AUDIO")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.smoke_test is not None:
        if args.output is None:
            parser.error("--smoke-test requires --output")
        return smoke_test(args.smoke_test, args.output)

    import tkinter as tk

    root = tk.Tk()
    MusicTaggingApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
