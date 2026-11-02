"""Offline selection of completed RECAP recordings; never opens hardware."""

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def labelled_runs(directory):
    result = []
    for path in sorted(Path(directory).glob("*/metadata.json"), reverse=True):
        try:
            metadata = json.loads(path.read_text())
            outcome = metadata.get("recap", {}).get("outcome")
            if outcome in ("success", "failure") and (path.parent / "trajectory.npz").is_file():
                result.append((path.parent, outcome))
        except (ValueError, OSError):
            continue
    return result


def choose_export(parent, directory, output_parent):
    result = []
    dialog = tk.Toplevel(parent)
    dialog.title("Export RECAP episodes")
    dialog.transient(parent)
    dialog.grab_set()
    body = ttk.Frame(dialog, padding=12)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="Select completed success/failure episodes (Ctrl/Shift for multiple). Aborted episodes are excluded.", wraplength=650).pack(anchor="w")
    source = tk.StringVar(value=str(directory))
    ttk.Label(body, textvariable=source).pack(anchor="w", pady=6)
    listing = tk.Listbox(body, selectmode="extended", width=85, height=16, exportselection=False)
    listing.pack(fill="both", expand=True)
    rows = []

    def refresh():
        rows[:] = labelled_runs(source.get())
        listing.delete(0, "end")
        for path, outcome in rows:
            listing.insert("end", f"{'Success' if outcome == 'success' else 'Failure'}  {path.name}")

    def browse():
        value = filedialog.askdirectory(parent=dialog, title="Recordings directory", initialdir=source.get())
        if value:
            source.set(value)
            refresh()

    def export():
        selected = [rows[i][0] for i in listing.curselection()]
        if not selected:
            messagebox.showinfo("Select episodes", "Select at least one labelled episode.", parent=dialog)
            return
        value = filedialog.asksaveasfilename(parent=dialog, title="New dataset directory (must not exist)",
                                            initialdir=str(output_parent), initialfile="recap_round_01")
        if not value:
            return
        if Path(value).exists():
            messagebox.showerror("Directory exists", "Choose a new dataset directory.", parent=dialog)
            return
        result.append((selected, Path(value)))
        dialog.destroy()

    actions = ttk.Frame(body)
    actions.pack(fill="x", pady=8)
    ttk.Button(actions, text="Change recordings directory", command=browse).pack(side="left")
    ttk.Button(actions, text="Export selected episodes", command=export).pack(side="right")
    refresh()
    parent.wait_window(dialog)
    return result[0] if result else None
