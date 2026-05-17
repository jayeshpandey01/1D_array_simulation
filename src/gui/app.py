"""Tkinter GUI: upload CSV, analyze signal, view report and plots."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Allow running as python src/gui/app.py from project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.constants import SAMPLE_RATE
from src.identify_amplitude_frequency import get_dominant_frequency
from src.pipeline import analyze_one
from src.signal_io import ground_truth_from_row, load_signal_csv
from src.signal_report import format_report, to_json


class SignalAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("1D Signal Parameter Analyzer")
        self.geometry("1000x720")
        self.minsize(800, 600)

        self.csv_path: str | None = None
        self.signals: np.ndarray | None = None
        self.metadata = None
        self.current_row = 0
        self.db_signal_id: int | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Open CSV…", command=self._open_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Analyze row", command=self._analyze).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Save report JSON", command=self._save_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Import CSV → DB", command=self._import_csv_to_db).pack(side=tk.LEFT, padx=4)

        db_frame = ttk.Frame(self, padding=(8, 0))
        db_frame.pack(fill=tk.X)
        ttk.Label(db_frame, text="DB signal id:").pack(side=tk.LEFT)
        self.db_id_spin = ttk.Spinbox(db_frame, from_=1, to=99999, width=8)
        self.db_id_spin.pack(side=tk.LEFT, padx=4)
        ttk.Button(db_frame, text="Load from DB", command=self._load_from_db).pack(side=tk.LEFT, padx=4)
        ttk.Button(db_frame, text="Analyze & save to DB", command=self._analyze_save_db).pack(side=tk.LEFT, padx=4)

        self.path_var = tk.StringVar(value="No file loaded")
        ttk.Label(top, textvariable=self.path_var).pack(side=tk.LEFT, padx=12)

        row_frame = ttk.Frame(self, padding=(8, 0))
        row_frame.pack(fill=tk.X)
        ttk.Label(row_frame, text="Row index:").pack(side=tk.LEFT)
        self.row_spin = ttk.Spinbox(row_frame, from_=0, to=0, width=8)
        self.row_spin.pack(side=tk.LEFT, padx=4)
        self.row_spin.bind("<Return>", lambda _: self._on_row_change())
        self.row_spin.bind("<FocusOut>", lambda _: self._on_row_change())

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(paned)
        paned.add(left, weight=2)
        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        ttk.Label(right, text="Analysis report").pack(anchor=tk.W)
        self.report_text = scrolledtext.ScrolledText(right, wrap=tk.WORD, font=("Consolas", 10))
        self.report_text.pack(fill=tk.BOTH, expand=True)

        self.last_result = None
        self._ground_truth_cache = None

    def _open_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select signal CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            signals, meta = load_signal_csv(path)
            self.csv_path = path
            self.signals = signals
            self.metadata = meta
            self.db_signal_id = None
            self._ground_truth_cache = None
            self.current_row = 0
            self.path_var.set(os.path.basename(path) + f"  ({signals.shape[0]} rows)")
            self.row_spin.config(to=max(0, signals.shape[0] - 1))
            self.row_spin.delete(0, tk.END)
            self.row_spin.insert(0, "0")
            self._plot_signal(signals[0])
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, f"Loaded {signals.shape[0]} signals.\nClick 'Analyze row'.")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _on_row_change(self) -> None:
        if self.signals is None:
            return
        try:
            idx = int(self.row_spin.get())
        except ValueError:
            return
        idx = max(0, min(idx, self.signals.shape[0] - 1))
        self.current_row = idx
        self._plot_signal(self.signals[idx])

    def _ground_truth_for_row(self, idx: int):
        if self.metadata is None or idx >= len(self.metadata):
            return None
        return ground_truth_from_row(self.metadata.iloc[idx])

    def _plot_signal(self, sig: np.ndarray) -> None:
        self.ax_time.clear()
        self.ax_freq.clear()
        self.ax_time.plot(sig)
        self.ax_time.set_title("Time domain")
        self.ax_time.set_xlabel("Sample")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.grid(True, alpha=0.3)

        freq, xf, yf = get_dominant_frequency(sig, SAMPLE_RATE)
        self.ax_freq.plot(xf, yf)
        self.ax_freq.set_title(f"FFT magnitude (peak ≈ {freq:.2f} Hz)")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_xlim(0, SAMPLE_RATE / 2)
        self.ax_freq.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def _analyze(self) -> None:
        if self.signals is None:
            messagebox.showinfo("Analyze", "Load a CSV file first.")
            return
        try:
            idx = int(self.row_spin.get())
        except ValueError:
            idx = self.current_row
        idx = max(0, min(idx, self.signals.shape[0] - 1))
        if self.db_signal_id is not None and self.signals.shape[0] == 1:
            gt = self._ground_truth_cache
        else:
            gt = self._ground_truth_for_row(idx)
        self.last_result = analyze_one(self.signals[idx], ground_truth=gt)
        text = format_report(self.last_result)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, text)
        self._plot_signal(self.signals[idx])

    def _import_csv_to_db(self) -> None:
        path = filedialog.askopenfilename(
            title="Import CSV into database",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        try:
            from src.database import import_csv, init_db

            init_db()
            n = import_csv(path)
            messagebox.showinfo("Database", f"Imported {n} signals into data/signals.db")
        except Exception as e:
            messagebox.showerror("Database import failed", str(e))

    def _load_from_db(self) -> None:
        try:
            from src.database import get_signal

            sid = int(self.db_id_spin.get())
            sig, gt = get_signal(sid)
            self.db_signal_id = sid
            self.signals = sig.reshape(1, -1)
            self.metadata = None
            self.csv_path = f"database://signal/{sid}"
            self.path_var.set(self.csv_path)
            self.row_spin.delete(0, tk.END)
            self.row_spin.insert(0, "0")
            self.row_spin.config(to=0)
            self._ground_truth_cache = gt
            self._plot_signal(sig)
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, f"Loaded signal id={sid} from database.\nClick Analyze row or Analyze & save to DB.")
        except Exception as e:
            messagebox.showerror("Load from DB failed", str(e))

    def _analyze_save_db(self) -> None:
        if self.db_signal_id is None:
            messagebox.showinfo("Database", "Load a signal from DB first (DB signal id → Load from DB).")
            return
        try:
            from src.database import analyze_and_store

            self.last_result = analyze_and_store(self.db_signal_id)
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, format_report(self.last_result))
            self.report_text.insert(tk.END, f"\n\n(Saved to analyses table, signal_id={self.db_signal_id})")
            self._plot_signal(self.signals[0])
        except Exception as e:
            messagebox.showerror("Analyze failed", str(e))

    def _save_json(self) -> None:
        if self.last_result is None:
            messagebox.showinfo("Save", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            to_json(self.last_result, path)
            messagebox.showinfo("Saved", f"Report saved to {path}")


def main() -> None:
    app = SignalAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
