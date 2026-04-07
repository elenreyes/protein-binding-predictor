#!/usr/bin/env python3
"""
Binding Site Predictor - GUI launcher
======================================
Run:  python binding_site_gui.py

Workflow:
  1. User picks a .pdb file via file-dialog
  2. Click "Run Prediction"
  3. predict.py logic runs in a background thread
  4. Results CSV + PyMOL script appear in results/
  5. PyMOL opens automatically with the coloured structure if installed
"""

import sys
import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import importlib.util

# ── Resolve paths relative to this script ────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
BORDER  = "#2a2d3a"
ACCENT  = "#4f9cf9"
ACCENT2 = "#7c6af7"
TEXT    = "#e8eaf0"
MUTED   = "#6b7280"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER  = "#f87171"
RED_HL  = "#ef4444"
ORANGE  = "#f97316"
YELLOW  = "#eab308"


def make_gradient_canvas(parent, w, h, c1, c2, vertical=True):
    """Draw a simple two-colour gradient on a Canvas."""
    cvs = tk.Canvas(parent, width=w, height=h, bd=0, highlightthickness=0, bg=BG)
    r1, g1, b1 = parent.winfo_rgb(c1)
    r2, g2, b2 = parent.winfo_rgb(c2)
    steps = h if vertical else w
    for i in range(steps):
        t  = i / steps
        r  = int(r1 + (r2 - r1) * t) >> 8
        g  = int(g1 + (g2 - g1) * t) >> 8
        b  = int(b1 + (b2 - b1) * t) >> 8
        col = f"#{r:02x}{g:02x}{b:02x}"
        if vertical:
            cvs.create_line(0, i, w, i, fill=col)
        else:
            cvs.create_line(i, 0, i, h, fill=col)
    return cvs


class BindingSiteApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Binding Site Predictor")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._center()

        self.pdb_path    = tk.StringVar()
        self.threshold   = tk.StringVar(value="")
        self.top_percent = tk.StringVar(value="0.12")
        self.running     = False

        self._build_ui()

    # ── Window helpers ────────────────────────────────────────────────────────

    def _center(self, w_ratio=0.8, h_ratio=0.8):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        w = int(sw * w_ratio)
        h = int(sh * h_ratio)

        x = (sw - w) // 2
        y = (sh - h) // 2

        self.geometry(f"{w}x{h}+{x}+{y}")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Header bar
        hdr = tk.Frame(self, bg=PANEL, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="Binding Site Predictor",
                 font=("Courier", 16, "bold"), fg=TEXT, bg=PANEL
                 ).pack(side="left", pady=14)

        # Separator
        tk.Frame(self, bg=ACCENT, height=2).pack(fill="x")

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=20)

        # ── PDB selector ──────────────────────────────────────────────────────
        self._section(body, "INPUT STRUCTURE")

        row_pdb = tk.Frame(body, bg=BG)
        row_pdb.pack(fill="x", pady=(4, 12))

        entry = tk.Entry(row_pdb, textvariable=self.pdb_path, width=56,
                         bg=PANEL, fg=TEXT, insertbackground=ACCENT,
                         relief="flat", font=("Courier", 10),
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT)
        entry.pack(side="left", ipady=6, padx=(0, 8))

        self._btn(row_pdb, "Browse…", self._browse, ACCENT).pack(side="left")

        # ── Parameters ───────────────────────────────────────────────────────
        self._section(body, "PARAMETERS")

        params = tk.Frame(body, bg=BG)
        params.pack(fill="x", pady=(4, 12))

        self._param(params, "Top %  (0-1)", self.top_percent, 0)
        self._param(params, "Threshold  (optional)", self.threshold, 1)

        # ── Legend ────────────────────────────────────────────────────────────
        self._section(body, "COLOUR LEGEND  -  PyMOL OUTPUT")
        leg = tk.Frame(body, bg=PANEL, bd=0, highlightthickness=1,
                       highlightbackground=BORDER)
        leg.pack(fill="x", pady=(4, 16), ipadx=12, ipady=10)

        for col, label, rng in [
            (RED_HL,  "High confidence",   "prob ≥ 0.70"),
            (ORANGE,  "Medium confidence", "0.50 ≤ prob < 0.70"),
            (YELLOW,  "Lower confidence",  "0.30 ≤ prob < 0.50"),
            ("#9ca3af","Not predicted",     "rest of protein"),
        ]:
            r = tk.Frame(leg, bg=PANEL)
            r.pack(anchor="w", pady=2)
            tk.Frame(r, bg=col, width=14, height=14).pack(side="left", padx=(0,8))
            tk.Label(r, text=label, fg=TEXT, bg=PANEL,
                     font=("Courier", 9, "bold")).pack(side="left")
            tk.Label(r, text=f"  {rng}", fg=MUTED, bg=PANEL,
                     font=("Courier", 9)).pack(side="left")

        # ── Run button ────────────────────────────────────────────────────────
        self.run_btn = self._btn(body, "▶  Run Prediction", self._run,
                                 ACCENT2, width=26, font_size=11)
        self.run_btn.pack(pady=(0, 12))

        # ── Log ───────────────────────────────────────────────────────────────
        self._section(body, "LOG")
        log_frame = tk.Frame(body, bg=PANEL, highlightthickness=1,
                             highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True)

        self.log = tk.Text(log_frame, bg=PANEL, fg=TEXT,
                           font=("Courier", 9), relief="flat",
                           insertbackground=ACCENT, wrap="word",
                           height=8, state="disabled")
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)

        # Tag colours for log levels
        self.log.tag_config("ok",   foreground=SUCCESS)
        self.log.tag_config("warn", foreground=WARNING)
        self.log.tag_config("err",  foreground=DANGER)
        self.log.tag_config("info", foreground=ACCENT)

        self._log("Ready. Select a PDB file and click Run Prediction.", "info")

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _section(self, parent, text):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=(6, 0))
        tk.Label(f, text=text, fg=MUTED, bg=BG,
                 font=("Courier", 8, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x",
                                               expand=True, padx=(8, 0))

    def _btn(self, parent, text, cmd, color, width=12, font_size=9):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg="white", activebackground=BG,
                      activeforeground=color, relief="flat", cursor="hand2",
                      font=("Courier", font_size, "bold"),
                      padx=14, pady=6, width=width)
        return b

    def _param(self, parent, label, var, col):
        f = tk.Frame(parent, bg=BG)
        f.grid(row=0, column=col, padx=(0, 24))
        tk.Label(f, text=label, fg=MUTED, bg=BG,
                 font=("Courier", 8)).pack(anchor="w")
        e = tk.Entry(f, textvariable=var, width=18,
                     bg=PANEL, fg=TEXT, insertbackground=ACCENT,
                     relief="flat", font=("Courier", 10),
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT)
        e.pack(ipady=5)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg, level=""):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n", level)
        self.log.see("end")
        self.log.configure(state="disabled")

    # ── File browser ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select PDB file",
            filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")]
        )
        if path:
            self.pdb_path.set(path)
            self._log(f"Selected: {path}", "ok")

    # ── Run prediction ────────────────────────────────────────────────────────

    def _run(self):
        if self.running:
            return

        pdb = self.pdb_path.get().strip()
        if not pdb or not Path(pdb).exists():
            messagebox.showerror("Error", "Please select a valid .pdb file.")
            return

        try:
            top = float(self.top_percent.get())
            assert 0 < top <= 1
        except Exception:
            messagebox.showerror("Error", "Top % must be a number between 0 and 1.")
            return

        thr_str = self.threshold.get().strip()
        thr = float(thr_str) if thr_str else None

        self.running = True
        self.run_btn.configure(state="disabled", text="⏳  Running…")
        self._log("─" * 55)
        self._log(f"PDB          : {pdb}", "info")
        self._log(f"Top %        : {top}", "info")
        if thr:
            self._log(f"Threshold    : {thr}", "info")
        self._log("Starting prediction …", "info")

        t = threading.Thread(target=self._worker,
                             args=(Path(pdb), thr, top), daemon=True)
        t.start()

    def _worker(self, pdb_path: Path, threshold, top_percent):
        """Runs in a background thread."""
        try:
            # Inject BASE_DIR into sys.path so predict.py / feature_engineering
            # can be imported even when launched from a different working dir.
            if str(BASE_DIR) not in sys.path:
                sys.path.insert(0, str(BASE_DIR))
            os.chdir(BASE_DIR)

            # Redirect stdout to our log widget
            import io
            old_stdout = sys.stdout
            sys.stdout  = _LogWriter(self._log)

            from predict import predict_binding_sites
            results = predict_binding_sites(
                pdb_path    = pdb_path,
                threshold   = threshold,
                top_percent = top_percent,
            )

            sys.stdout = old_stdout

            # ── Open PyMOL ────────────────────────────────────────────────────
            pml_file = RESULTS_DIR / f"predictions_{pdb_path.stem}.pml"
            if pml_file.exists():
                self._log(f"\nOpening PyMOL with {pml_file.name} …", "ok")
                self.after(0, lambda: self._open_pymol(pml_file))
            else:
                self._log("PyMOL script not found - skipping.", "warn")

            n_pred = int(results["predicted"].sum())
            total  = len(results)
            self._log(f"\n✓ Done - {n_pred}/{total} residues predicted as binding site.", "ok")
            self._log(f"  CSV : {RESULTS_DIR / f'predictions_{pdb_path.stem}.csv'}", "ok")
            self._log(f"  PML : {pml_file}", "ok")

        except Exception as exc:
            self._log(f"\n✗ Error: {exc}", "err")
            import traceback
            self._log(traceback.format_exc(), "err")
        finally:
            self.running = False
            self.after(0, lambda: self.run_btn.configure(
                state="normal", text="▶  Run Prediction"))

    def _open_pymol(self, pml_file: Path):
        """Try to launch PyMOL with the generated script."""
        pymol_candidates = [
            "pymol",                              # on PATH
            "/usr/bin/pymol",
            "/usr/local/bin/pymol",
            r"C:\Program Files\PyMOL\PyMOL\PyMOL.exe",
            r"C:\Program Files (x86)\PyMOL\PyMOL\PyMOL.exe",
            "/Applications/PyMOL.app/Contents/MacOS/PyMOL",
            r"C:\Users\nuria\anaconda3\envs\pymol-env\Scripts\pymol.exe",
        ] # Add own PATH if not already in
        for exe in pymol_candidates:
            try:
                subprocess.Popen([exe, str(pml_file)])
                self._log(f"  PyMOL launched: {exe}", "ok")
                return
            except FileNotFoundError:
                continue

        self._log("  PyMOL not found on PATH. Open the .pml file manually.", "warn")
        # Offer to reveal the file
        try:
            if sys.platform == "win32":
                os.startfile(str(RESULTS_DIR))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(RESULTS_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(RESULTS_DIR)])
        except Exception:
            pass


class _LogWriter:
    """Redirect stdout prints to the Tkinter log widget."""
    def __init__(self, log_fn):
        self._log = log_fn
    def write(self, s):
        s = s.rstrip("\n")
        if s:
            self._log(s)
    def flush(self):
        pass


if __name__ == "__main__":
    app = BindingSiteApp()
    app.mainloop()
