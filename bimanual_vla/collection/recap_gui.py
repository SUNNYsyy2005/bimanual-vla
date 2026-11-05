"""Offline selection of completed RECAP recordings; never opens hardware."""

import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def _pending_rollout(directory):
    if not str(directory or "").strip():
        raise ValueError("No rollout recording is available for this session.")
    path = Path(directory)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    recap = metadata.get("recap", {})
    if not metadata.get("finished_at") or not (path / "trajectory.npz").is_file():
        raise ValueError("The rollout has not finished saving; its outcome cannot be changed yet.")
    if (not recap.get("finished") or recap.get("error")
            or not recap.get("label_requested") or recap.get("outcome") != "unlabelled"):
        raise ValueError("This rollout is not waiting for an outcome label.")
    return path, metadata


def label_completed_rollout(directory, outcome):
    """Only label a finalized A-stopped rollout; never rewrite trajectory data."""
    if outcome not in ("success", "failure", "aborted"):
        raise ValueError("Choose Success, Failure, or Abort.")
    from bimanual_vla.deployment.recap import atomic_json
    path, metadata = _pending_rollout(directory)
    metadata["recap"].update(outcome=outcome, label_requested=False,
                             labelled_at=time.time(), label_source="vr_a_dialog")
    atomic_json(path / "metadata.json", metadata)


def choose_rollout_outcome(parent, directory, *, show_error):
    """Closing the English outcome dialog selects Abort, preserving the run."""
    path, _ = _pending_rollout(directory)
    result = []
    dialog = tk.Toplevel(parent)
    dialog.title("Rollout outcome")
    dialog.transient(parent)
    dialog.resizable(False, False)
    body = ttk.Frame(dialog, padding=16)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="Inference stopped. Choose the outcome for this rollout.").pack(anchor="w")
    ttk.Label(body, text=path.name).pack(anchor="w", pady=(6, 12))
    actions = ttk.Frame(body)
    actions.pack(fill="x")

    def finish(outcome):
        try:
            label_completed_rollout(path, outcome)
        except (OSError, ValueError) as exc:
            show_error("Cannot save rollout outcome", str(exc), parent=dialog)
            return
        result.append(outcome)
        dialog.destroy()

    for outcome, label in (("success", "Success"), ("failure", "Failure"), ("aborted", "Abort")):
        ttk.Button(actions, text=label, command=lambda value=outcome: finish(value)).pack(side="left", padx=4)
    dialog.protocol("WM_DELETE_WINDOW", lambda: finish("aborted"))
    dialog.bind("<Escape>", lambda _event: finish("aborted"))
    dialog.grab_set()
    parent.wait_window(dialog)
    return result[0] if result else None


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
