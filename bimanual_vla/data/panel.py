"""Tk read-only Data process page for collected robot data."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk
from typing import Callable, Iterable

import numpy as np

from bimanual_vla.data.analysis import (
    AnalysisData,
    compute_metrics,
    compute_end_effector_positions,
    jitter_plot_series,
    load_analysis_data,
    motion_joint_names,
    scan_analysis_sources,
    selection_indices,
    policy_trajectory_series,
    sent_chunk_switch_indices,
    trajectory_chunk_switch_times,
    trajectory_motion_series,
)
from bimanual_vla.data.eef_view import draw_end_effector_trajectory


def _deployment_joint_values(measured: np.ndarray) -> np.ndarray:
    """Drop per-arm gripper columns to match ``command_joints_rad``."""
    values = np.asarray(measured, dtype=np.float64)
    if values.ndim != 2:
        return np.empty((0, 0), dtype=np.float64)
    if values.shape[1] == 7:
        return values[:, :6]
    if values.shape[1] == 14:
        return np.concatenate((values[:, :6], values[:, 7:13]), axis=1)
    return values


def _deployment_joint_index(index: int, width: int) -> int | None:
    """Map state-column indices (which include grippers) to motor columns."""
    if width == 7:
        return index if 0 <= index < 6 else None
    if width == 14:
        if 0 <= index < 6:
            return index
        if 7 <= index < 13:
            return index - 1
        return None
    return index if index >= 0 else None


MOTION_VIEWS = {
    "Velocity": (1, "rad/s"),
    "Acceleration": (2, "rad/s²"),
}

CHUNK_VIEWS = {
    "Boundary position jump": ("boundary_jump_mean_rad_l2", "rad L2"),
    "Boundary velocity direction": ("boundary_momentum_cosine_mean", "cosine"),
}

PLOT_GROUPS = {
    "Trajectory": ("Position", *MOTION_VIEWS, "Tracking error"),
    "Chunk boundaries": tuple(CHUNK_VIEWS),
    "Timing": ("Inference latency", "Control interval"),
    "End effector": ("3D trajectory",),
}
TRAJECTORY_VIEWS = frozenset(PLOT_GROUPS["Trajectory"])


def available_plot_groups(data: AnalysisData | None) -> dict[str, tuple[str, ...]]:
    """Only offer charts backed by the selected source's recording format."""
    if data is not None and data.kind == "episode":
        return {
            "Trajectory": ("Position", *MOTION_VIEWS, "Tracking error"),
            "Timing": ("Control interval",),
            "End effector": PLOT_GROUPS["End effector"],
        }
    return PLOT_GROUPS


def _episode_motion_series(
    times: np.ndarray,
    values: np.ndarray,
    *,
    order: int,
    joint_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate episode joint signals against their recorded timestamps."""
    if order not in (1, 2) or len(times) != len(values) or len(times) < order + 1:
        return np.empty(0, dtype=np.float64), np.empty((0, len(joint_indices)), dtype=np.float64)

    dt = np.diff(times)
    valid_dt = np.isfinite(dt) & (dt > 0)
    selected = values[:, joint_indices]
    valid_values = np.isfinite(selected[:-1]) & np.isfinite(selected[1:])
    velocity = np.full((len(dt), len(joint_indices)), np.nan, dtype=np.float64)
    valid = valid_values & valid_dt[:, np.newaxis]
    np.divide(np.diff(selected, axis=0), dt[:, np.newaxis], out=velocity, where=valid)
    velocity_times = (times[:-1] + times[1:]) / 2.0
    if order == 1:
        return velocity_times, velocity

    acceleration_dt = np.diff(velocity_times)
    valid_acceleration_dt = np.isfinite(acceleration_dt) & (acceleration_dt > 0)
    acceleration = np.full((len(acceleration_dt), len(joint_indices)), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(velocity[:-1])
        & np.isfinite(velocity[1:])
        & valid_acceleration_dt[:, np.newaxis]
    )
    np.divide(
        np.diff(velocity, axis=0),
        acceleration_dt[:, np.newaxis],
        out=acceleration,
        where=valid,
    )
    return (velocity_times[:-1] + velocity_times[1:]) / 2.0, acceleration


def _timing_value_ms(record: dict[str, object], key: str) -> float:
    """Represent missing or invalid per-request diagnostics as a chart gap."""
    timing = record.get("_client_transport_timing")
    if not isinstance(timing, dict):
        return float("nan")
    try:
        value = float(timing.get(key))
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _ask_english_yes_no(parent: tk.Misc, title: str, message: str) -> bool:
    """Show a confirmation dialog with English button labels."""
    result = {"confirmed": False}
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()
    body = ttk.Frame(dialog, padding=16)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text=message, justify="left", wraplength=520).pack(anchor="w", fill="x")
    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(16, 0))
    buttons.columnconfigure(0, weight=1)

    def finish(confirmed: bool) -> None:
        result["confirmed"] = confirmed
        dialog.grab_release()
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=lambda: finish(False)).grid(row=0, column=1, padx=(0, 8))
    ttk.Button(buttons, text="Move to trash", command=lambda: finish(True)).grid(row=0, column=2)
    dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
    parent.wait_window(dialog)
    return bool(result["confirmed"])


class DataProcessPanel(ttk.Frame):
    """Interactive, non-destructive analysis view.

    The panel stores only the selected time range and source path.  It never
    writes to the source files.
    """

    def __init__(self, parent: tk.Misc, roots_provider: Callable[[], Iterable[Path]]) -> None:
        super().__init__(parent, padding=24)
        families = set(tkfont.families(parent.winfo_toplevel()))
        self.font_name = "Times New Roman" if "Times New Roman" in families else ("Liberation Serif" if "Liberation Serif" in families else "DejaVu Serif")
        self.roots_provider = roots_provider
        self.sources: list[Path] = []
        self.displayed_sources: list[Path] = []
        self.data: AnalysisData | None = None
        self.pose_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._scan_request_id = 0
        self._load_request_id = 0
        self.start_var = tk.StringVar(value="0.00")
        self.end_var = tk.StringVar(value="0.00")
        self.source_filter_var = tk.StringVar(value="Run")
        self.start_label_var = tk.StringVar(value="0.00 s")
        self.end_label_var = tk.StringVar(value="0.00 s")
        self._range_update_guard = False
        self.group_var = tk.StringVar(value="Trajectory")
        self.plot_var = tk.StringVar(value="Velocity")
        self.signal_var = tk.StringVar(value="")
        self.selection_var = tk.StringVar(value="No data selected")
        self.status_var = tk.StringVar(value="Select a data source")
        self.metric_vars: dict[str, tk.StringVar] = {
            key: tk.StringVar(value="—")
            for key in (
                "source",
                "duration",
                "control",
                "inference",
                "latency",
                "jitter",
                "sent",
                "action_rows",
                "executed_actions",
                "rejected_rows",
                "unsafe_events",
                "discarded_rows",
            )
        }
        self.chart: tk.Canvas | None = None
        self.chart_series: list[tuple[str, np.ndarray, str]] = []
        self.chart_x = np.array([], dtype=np.float64)
        self.chart_y_label = ""
        self.chart_markers: list[tuple[str, np.ndarray, str]] = []
        self.chart_series_x: list[np.ndarray] = []
        self.chart_series_dash_patterns: list[tuple[int, ...] | None] = []
        self.chart_series_visible: dict[str, bool] = {}
        self.chart_legend_regions: list[tuple[float, float, float, float, str]] = []
        self.chart_legend_vars: dict[str, tk.BooleanVar] = {}
        self.chart_legend_widgets: dict[str, tk.Checkbutton] = {}
        self.eef_yaw = 0.0
        self.eef_pitch = 0.45
        self.eef_zoom = 1.0
        self.eef_drag_at: tuple[int, int] | None = None
        self._build_ui()
        self.refresh_sources()

    def _build_ui(self) -> None:
        analysis_style = ttk.Style(self)
        analysis_style.configure(
            "Analysis.Treeview",
            rowheight=38,
            font=(self.font_name, 10),
            padding=(4, 3),
        )
        analysis_style.configure(
            "Analysis.Treeview.Heading",
            font=(self.font_name, 10, "bold"),
            padding=(6, 9),
        )
        self.columnconfigure(0, weight=0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        filter_bar = ttk.Frame(left)
        filter_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        filter_bar.columnconfigure(1, weight=1)
        ttk.Label(filter_bar, text="Data source").grid(row=0, column=0, sticky="w")
        self.source_filter_selector = ttk.Combobox(
            filter_bar, textvariable=self.source_filter_var,
            values=("Run", "Episode", "All"), state="readonly", width=12,
        )
        self.source_filter_selector.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.source_filter_selector.bind("<<ComboboxSelected>>", lambda _event: self._render_source_tree())
        ttk.Button(filter_bar, text="Refresh", command=self.refresh_sources).grid(row=0, column=2)
        source_box = ttk.LabelFrame(left, text="Data sources", padding=8)
        source_box.grid(row=1, column=0, sticky="nsew")
        source_box.rowconfigure(0, weight=1)
        source_box.columnconfigure(0, weight=1)
        self.source_tree = ttk.Treeview(
            source_box, columns=("kind", "source"), show="headings", height=18,
            style="Analysis.Treeview",
        )
        self.source_tree.heading("kind", text="Type")
        self.source_tree.heading("source", text="Source")
        self.source_tree.column("kind", width=90, minwidth=70, stretch=False)
        self.source_tree.column("source", width=250, minwidth=130, stretch=True)
        self.source_tree.grid(row=0, column=0, sticky="nsew")
        source_scroll = ttk.Scrollbar(source_box, orient="vertical", command=self.source_tree.yview)
        source_scroll.grid(row=0, column=1, sticky="ns")
        self.source_tree.configure(yscrollcommand=source_scroll.set)
        self.source_tree.bind("<<TreeviewSelect>>", self._source_selected)
        ttk.Label(left, textvariable=self.status_var, foreground="#68707d", wraplength=300).grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Button(left, text="Move selected data to trash", command=self._delete_selected_source).grid(
            row=3, column=0, sticky="ew", pady=(8, 0)
        )

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(right)
        notebook.grid(row=0, column=0, sticky="nsew")

        overview = ttk.Frame(notebook, padding=(0, 0, 0, 4))
        overview.columnconfigure(0, weight=1)
        overview.rowconfigure(1, weight=1)
        notebook.add(overview, text="Overview")

        charts = ttk.Frame(notebook, padding=(0, 0, 0, 4))
        charts.columnconfigure(0, weight=1)
        charts.rowconfigure(2, weight=1)
        notebook.add(charts, text="Charts")

        summary = ttk.LabelFrame(overview, text="Summary", padding=10)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        summary.rowconfigure(0, minsize=58)
        summary.rowconfigure(1, minsize=58)
        for column in range(4):
            summary.columnconfigure(column, weight=1)
        cards = (
            ("source", "Source"), ("duration", "Selected duration"),
            ("control", "Control rate"), ("inference", "Model commands"),
            ("latency", "Round-trip P50 / P95"), ("jitter", "Tick jitter P95"),
            ("sent", "Command coverage"),
        )
        for index, (key, title) in enumerate(cards):
            row, column = divmod(index, 4)
            card = tk.Frame(summary, bg="#f7f8fa", highlightthickness=1, highlightbackground="#e4e7ec")
            card.grid(row=row, column=column, sticky="ew", padx=4, pady=4)
            tk.Label(card, text=title, bg="#f7f8fa", fg="#68707d", font=("Liberation Serif", 9), anchor="w").pack(fill="x", padx=8, pady=(6, 1))
            tk.Label(card, textvariable=self.metric_vars[key], bg="#f7f8fa", fg="#202124", font=("Liberation Serif", 10, "bold"), anchor="w").pack(fill="x", padx=8, pady=(0, 6))

        overview_body = ttk.Frame(overview)
        overview_body.grid(row=1, column=0, sticky="nsew")
        overview_body.columnconfigure(0, weight=3, uniform="overview-column")
        overview_body.columnconfigure(1, weight=2, uniform="overview-column")
        overview_body.rowconfigure(0, weight=1)

        accounting = ttk.LabelFrame(overview_body, text="Actions and chunk metrics", padding=8)
        accounting.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        for column in range(4):
            accounting.columnconfigure(column, weight=1)
        accounting.columnconfigure(4, weight=0)
        accounting.rowconfigure(0, minsize=58)
        accounting.rowconfigure(1, minsize=58)
        accounting.rowconfigure(2, minsize=32)
        accounting.rowconfigure(3, weight=1)
        accounting_cards = (
            ("action_rows", "Rows returned by model"),
            ("executed_actions", "Control actions sent"),
            ("rejected_rows", "Rejected action rows"),
            ("unsafe_events", "Unsafe drop events"),
            ("discarded_rows", "Discarded rows (estimated)"),
        )
        for index, (key, title) in enumerate(accounting_cards):
            row, column = divmod(index, 4)
            card = tk.Frame(accounting, bg="#f7f8fa", highlightthickness=1, highlightbackground="#e4e7ec")
            card.grid(row=row, column=column, sticky="nsew", padx=4, pady=5, ipady=4)
            tk.Label(card, text=title, bg="#f7f8fa", fg="#68707d", font=("Liberation Serif", 9), anchor="w").pack(fill="x", padx=8, pady=(5, 1))
            tk.Label(card, textvariable=self.metric_vars[key], bg="#f7f8fa", fg="#202124", font=("Liberation Serif", 10, "bold"), anchor="w").pack(fill="x", padx=8, pady=(0, 5))
        ttk.Label(
            accounting,
            text="Trajectory jitter · policy output uses the full horizon; sent uses issued joint commands",
            foreground="#68707d",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(5, 2))
        self.jitter_tree = ttk.Treeview(
            accounting, columns=("metric", "model", "sent"), show="headings", height=8,
            style="Analysis.Treeview",
        )
        for key, title, width in (
            ("metric", "Metric", 240),
            ("model", "Policy output", 170),
            ("sent", "Command sent", 170),
        ):
            self.jitter_tree.heading(key, text=title)
            self.jitter_tree.column(key, width=width, minwidth=110, stretch=True)
        self.jitter_tree.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=(4, 0), pady=(0, 4))
        jitter_scroll = ttk.Scrollbar(accounting, orient="vertical", command=self.jitter_tree.yview)
        jitter_scroll.grid(row=3, column=4, sticky="ns", padx=(4, 4), pady=(0, 4))
        self.jitter_tree.configure(yscrollcommand=jitter_scroll.set)

        controls = ttk.LabelFrame(charts, text="Analysis range", padding=8)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)
        ttk.Label(controls, text="Start").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.start_scale = ttk.Scale(controls, from_=0.0, to=1.0, orient="horizontal", command=self._range_changed)
        self.start_scale.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(controls, textvariable=self.start_label_var, width=10).grid(row=0, column=2, padx=(0, 18))
        ttk.Label(controls, text="End").grid(row=0, column=3, sticky="w", padx=(0, 8))
        self.end_scale = ttk.Scale(controls, from_=0.0, to=1.0, orient="horizontal", command=self._range_changed)
        self.end_scale.grid(row=0, column=4, sticky="ew", padx=(0, 8))
        ttk.Label(controls, textvariable=self.end_label_var, width=10).grid(row=0, column=5, padx=(0, 12))
        ttk.Button(controls, text="Full range", command=self._full_range).grid(row=0, column=6)
        ttk.Label(controls, textvariable=self.selection_var, foreground="#68707d").grid(row=1, column=0, columnspan=7, sticky="w", pady=(6, 0))

        chart_controls = ttk.Frame(charts)
        chart_controls.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(chart_controls, text="Group").pack(side="left")
        self.group_selector = ttk.Combobox(
            chart_controls, textvariable=self.group_var, state="readonly",
            values=tuple(PLOT_GROUPS), width=15,
        )
        self.group_selector.pack(side="left", padx=(8, 12))
        self.group_selector.bind("<<ComboboxSelected>>", lambda _event: self._group_changed())
        ttk.Label(chart_controls, text="View").pack(side="left")
        self.plot_selector = ttk.Combobox(
            chart_controls, textvariable=self.plot_var, state="readonly",
            values=PLOT_GROUPS["Trajectory"], width=27,
        )
        self.plot_selector.pack(side="left", padx=(8, 12))
        self.plot_selector.bind("<<ComboboxSelected>>", lambda _event: self._plot_changed())
        ttk.Label(chart_controls, text="Signal").pack(side="left")
        self.signal_selector = ttk.Combobox(chart_controls, textvariable=self.signal_var, state="readonly", values=("",), width=18)
        self.signal_selector.pack(side="left", padx=8)
        self.signal_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_chart())
        self.reset_3d_button = ttk.Button(chart_controls, text="Reset 3D view", command=self._reset_3d_view)

        self.chart = tk.Canvas(charts, background="#ffffff", highlightthickness=1, highlightbackground="#d9dde5")
        self.chart.grid(row=2, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())
        self.chart.bind("<ButtonPress-1>", self._eef_drag_start)
        self.chart.bind("<B1-Motion>", self._eef_drag_move)
        self.chart.bind("<ButtonRelease-1>", self._eef_drag_end)
        self.chart.bind("<ButtonRelease-1>", self._chart_legend_click, add="+")
        self.chart.bind("<MouseWheel>", self._eef_mouse_wheel)
        self.chart.bind("<Button-4>", self._eef_mouse_wheel)
        self.chart.bind("<Button-5>", self._eef_mouse_wheel)
        self.legend_frame = ttk.Frame(charts)
        self.legend_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        for column in range(3):
            self.legend_frame.columnconfigure(column, weight=1)

        events = ttk.LabelFrame(overview_body, text="Events in selected range", padding=6)
        events.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        events.columnconfigure(0, weight=1)
        events.rowconfigure(0, weight=1)
        self.events_tree = ttk.Treeview(
            events, columns=("event", "count"), show="headings", height=10,
            style="Analysis.Treeview",
        )
        self.events_tree.heading("event", text="Event")
        self.events_tree.heading("count", text="Count")
        self.events_tree.column("event", width=520, anchor="w")
        self.events_tree.column("count", width=80, anchor="center", stretch=False)
        self.events_tree.grid(row=0, column=0, sticky="nsew")
        events_scroll = ttk.Scrollbar(events, orient="vertical", command=self.events_tree.yview)
        events_scroll.grid(row=0, column=1, sticky="ns")
        self.events_tree.configure(yscrollcommand=events_scroll.set)

    def refresh_sources(self) -> None:
        current = self._selected_path()
        self._scan_request_id += 1
        request_id = self._scan_request_id
        self.status_var.set("Scanning data sources…")
        roots = tuple(self.roots_provider())

        def scan() -> None:
            try:
                sources = scan_analysis_sources(roots)
                error = None
            except Exception as exc:  # keep refresh non-fatal
                sources, error = [], exc

            def finish() -> None:
                if request_id != self._scan_request_id:
                    return
                self.sources = sources
                if error is not None:
                    self.status_var.set(f"Unable to scan data: {error}")
                self._render_source_tree(current)

            try:
                self.after(0, finish)
            except tk.TclError:
                pass

        threading.Thread(target=scan, name="data-source-scan", daemon=True).start()

    def _render_source_tree(self, current: Path | None = None) -> None:
        if current is None:
            current = self._selected_path()
        for item in self.source_tree.get_children():
            self.source_tree.delete(item)
        selected_iid = None
        filter_value = self.source_filter_var.get()
        self.displayed_sources = [
            path for path in self.sources
            if filter_value == "All"
            or (filter_value == "Run" and path.is_dir())
            or (filter_value == "Episode" and path.is_file())
        ]
        for index, path in enumerate(self.displayed_sources):
            kind = "Run" if path.is_dir() else "Episode"
            label = path.name if path.is_dir() else f"{path.parent.name}/{path.name}"
            iid = str(index)
            self.source_tree.insert("", "end", iid=iid, values=(kind, label))
            if current is not None and path == current:
                selected_iid = iid
        if selected_iid is not None:
            self.source_tree.selection_set(selected_iid)
            self.source_tree.focus(selected_iid)
            selected_path = self.displayed_sources[int(selected_iid)]
            if self.data is None or self.data.path != selected_path.resolve():
                self._load_source(selected_path)
        elif self.displayed_sources:
            self.source_tree.selection_set("0")
            self.source_tree.focus("0")
            self._load_source(self.displayed_sources[0])
        else:
            self.data = None
            self.status_var.set("No deployment runs or ep_XXXX.npz files found")
            self._clear_view()

    def _selected_path(self) -> Path | None:
        selection = self.source_tree.selection()
        if not selection:
            return None
        try:
            return self.displayed_sources[int(selection[0])]
        except (ValueError, IndexError):
            return None

    def _source_selected(self, _event=None) -> None:
        path = self._selected_path()
        if path is not None:
            self._load_source(path)

    def _load_source(self, path: Path) -> None:
        self._load_request_id += 1
        request_id = self._load_request_id
        self.data = None
        self.status_var.set(f"Loading {path.name}…")
        self._clear_view()

        def load() -> None:
            try:
                data = load_analysis_data(path)
                error = None
            except Exception as exc:
                data, error = None, exc

            def finish() -> None:
                if request_id != self._load_request_id or path not in self.displayed_sources:
                    return
                if error is not None:
                    self.data = None
                    self.status_var.set(f"Cannot load {path.name}: {error}")
                    self._clear_view()
                    return
                self._apply_loaded_source(data)

            try:
                self.after(0, finish)
            except tk.TclError:
                pass

        threading.Thread(target=load, name="analysis-data-load", daemon=True).start()

    def _apply_loaded_source(self, data: AnalysisData) -> None:
        self.data = data
        self.pose_cache.pop(data.path, None)
        self.start_scale.configure(to=max(0.01, data.duration_s))
        self.end_scale.configure(to=max(0.01, data.duration_s))
        self.start_scale.set(0.0)
        self.end_scale.set(data.duration_s)
        self.start_var.set("0.00")
        self.end_var.set(f"{data.duration_s:.2f}")
        self._sync_plot_options()
        self.status_var.set(f"Loaded {data.label} · {data.sample_count} samples")
        self.apply_range()

    def _delete_selected_source(self) -> None:
        path = self._selected_path()
        if path is None or path not in self.displayed_sources:
            messagebox.showinfo("Move data to trash", "Select a data source first.", parent=self)
            return
        if not _ask_english_yes_no(
            self,
            "Move data to trash",
            f"Move this data source to a recoverable .trash folder?\n\n{path}",
        ):
            return
        try:
            path = path.resolve(strict=True)
            if path not in {source.resolve() for source in self.sources}:
                raise ValueError("Selected path is no longer a recognized data source")
            trash_root = path.parent / ".trash"
            stamp = time.strftime("%Y%m%d-%H%M%S")
            trash_dir = trash_root / stamp
            suffix = 1
            while trash_dir.exists():
                trash_dir = trash_root / f"{stamp}-{suffix}"
                suffix += 1
            trash_dir.mkdir(parents=True)
            destination = trash_dir / path.name
            path.rename(destination)
        except Exception as exc:
            messagebox.showerror("Move failed", str(exc), parent=self)
            return
        self._load_request_id += 1
        self._scan_request_id += 1
        self.data = None
        self.pose_cache.pop(path, None)
        self._clear_view()
        self.status_var.set(f"Moved to {destination}")
        self.refresh_sources()

    def _clear_view(self) -> None:
        for variable in self.metric_vars.values():
            variable.set("—")
        self.selection_var.set("No data selected")
        self.chart_series = []
        self.chart_x = np.array([], dtype=np.float64)
        self.chart_markers = []
        self.chart_series_x = []
        self.chart_series_visible = {}
        self.chart_legend_regions = []
        self._render_chart_legend()
        if self.chart is not None:
            self.chart.delete("all")
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
        for item in self.jitter_tree.get_children():
            self.jitter_tree.delete(item)

    def _range_changed(self, _value=None) -> None:
        if self.data is None or self._range_update_guard:
            return
        start = float(self.start_scale.get())
        end = float(self.end_scale.get())
        if start > end:
            if _value is not None and abs(float(_value) - start) < 1e-9:
                self.end_scale.set(start)
                end = start
            else:
                self.start_scale.set(end)
                start = end
        self.start_var.set(f"{start:.2f}")
        self.end_var.set(f"{end:.2f}")
        self.start_label_var.set(f"{start:.2f} s")
        self.end_label_var.set(f"{end:.2f} s")
        self.apply_range()

    def _full_range(self) -> None:
        if self.data is None:
            return
        self.start_scale.set(0.0)
        self.end_scale.set(self.data.duration_s)
        self._range_changed()

    def apply_range(self) -> None:
        if self.data is None:
            return
        try:
            start_s = float(self.start_var.get())
            end_s = float(self.end_var.get())
        except ValueError:
            messagebox.showerror("Invalid range", "The selected range is invalid.", parent=self)
            return
        start, end = selection_indices(self.data, start_s, end_s)
        self.start_var.set(f"{self.data.timestamps[start] - self.data.timestamps[0]:.2f}")
        self.end_var.set(f"{self.data.timestamps[end] - self.data.timestamps[0]:.2f}")
        self.start_label_var.set(f"{self.start_var.get()} s")
        self.end_label_var.set(f"{self.end_var.get()} s")
        self._range_update_guard = True
        try:
            self.start_scale.set(float(self.start_var.get()))
            self.end_scale.set(float(self.end_var.get()))
        finally:
            self._range_update_guard = False
        metrics = compute_metrics(self.data, start, end)
        self.selection_var.set(
            f"Samples {start + 1}–{end + 1} · {metrics['start_s']:.2f}–{metrics['end_s']:.2f}s"
        )
        self._update_summary(metrics)
        self._update_events(metrics)
        self._refresh_chart()

    def _update_summary(self, metrics: dict[str, object]) -> None:
        latency = metrics.get("latency") or {}
        round_trip = latency.get("round_trip_ms") or {}
        tick = metrics.get("tick_interval_ms") or {}
        self.metric_vars["source"].set(self.data.kind if self.data else "—")
        self.metric_vars["duration"].set(f"{float(metrics.get('duration_s') or 0):.2f} s")
        control = metrics.get("control_hz")
        self.metric_vars["control"].set("—" if control is None else f"{float(control):.2f} Hz")
        self.metric_vars["inference"].set(str(metrics.get("model_command_count") or 0))
        p50, p95 = round_trip.get("median"), round_trip.get("p95")
        self.metric_vars["latency"].set("—" if p50 is None else f"{p50:.0f} / {p95:.0f} ms")
        jitter = tick.get("p95")
        self.metric_vars["jitter"].set("—" if jitter is None else f"{jitter:.1f} ms")
        self.metric_vars["sent"].set(f"{float(metrics.get('command_sent_fraction') or 0) * 100:.1f}%")
        unsafe = int(metrics.get("unsafe_drop_count") or 0)
        discarded = int(metrics.get("discarded_action_count") or 0)
        rejected_rows = int(metrics.get("rejected_action_rows") or 0)
        self.metric_vars["action_rows"].set(str(metrics.get("model_action_rows") or 0))
        self.metric_vars["executed_actions"].set(str(metrics.get("executed_control_actions") or 0))
        self.metric_vars["rejected_rows"].set(str(rejected_rows))
        self.metric_vars["unsafe_events"].set(str(unsafe))
        self.metric_vars["discarded_rows"].set(str(discarded))
        for item in self.jitter_tree.get_children():
            self.jitter_tree.delete(item)
        jitter_metrics = metrics.get("trajectory_jitter") or {}
        model = jitter_metrics.get("model_raw") or {}
        sent_jitter = jitter_metrics.get("command_sent") or {}
        for label, value_key, count_key in (
            ("Intra accel (rad/step²)", "intra_accel_mean_rad_per_step2", "intra_accel_samples"),
            ("Boundary jump (rad)", "boundary_jump_mean_rad_l2", "boundary_jump_samples"),
            ("Boundary velocity cosine", "boundary_momentum_cosine_mean", "boundary_momentum_samples"),
        ):
            def display(values):
                value = values.get(value_key)
                count = int(values.get(count_key) or 0)
                return "—" if value is None else f"{float(value):.4f} (n={count})"
            self.jitter_tree.insert("", "end", values=(label, display(model), display(sent_jitter)))

    def _update_events(self, metrics: dict[str, object]) -> None:
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
        events = Counter()
        for key, value in (metrics.get("blocked") or {}).items():
            events[key] += int(value)
        for key, value in (metrics.get("execution_states") or {}).items():
            if key and key not in {"executing", "recorded"}:
                events[f"state: {key}"] += int(value)
        if self.data is not None:
            start, end = self._selected_indices()
            if end >= start:
                events["action queue hold"] += int(np.count_nonzero(self.data.command_hold[start : end + 1]))
        for event, count in events.most_common():
            self.events_tree.insert("", "end", values=(event, count))

    def _selected_indices(self) -> tuple[int, int]:
        if self.data is None:
            return 0, -1
        try:
            start_s, end_s = float(self.start_var.get()), float(self.end_var.get())
        except ValueError:
            return 0, self.data.sample_count - 1
        return selection_indices(self.data, start_s, end_s)

    def _refresh_chart(self) -> None:
        if self.data is None:
            self.chart_series = []
            self.chart_x = np.array([], dtype=np.float64)
            self.chart_markers = []
            self.chart_series_x = []
            self.chart_series_visible = {}
            self._render_chart_legend()
            self._draw_chart()
            return
        start, end = self._selected_indices()
        if end < start:
            return
        if self.plot_var.get() == "3D trajectory":
            self.chart_series = []
            self.chart_x = np.array([], dtype=np.float64)
            self.chart_markers = []
            self.chart_series_x = []
            self.chart_series_visible = {}
            self._render_chart_legend()
            self._draw_chart()
            return
        self.chart_series, self.chart_x, self.chart_y_label = self._make_series(start, end)
        previous = getattr(self, "chart_series_visible", {})
        self.chart_series_visible = {
            label: previous.get(label, True)
            for label, _values, _color in self.chart_series
        }
        self._render_chart_legend()
        self._draw_chart()

    def _chart_legend_click(self, event: tk.Event) -> None:
        """Toggle one plotted series when its legend entry is clicked."""
        if self.plot_var.get() == "3D trajectory":
            return
        x, y = float(event.x), float(event.y)
        for left, top, right, bottom, label in getattr(self, "chart_legend_regions", []):
            if left <= x <= right and top <= y <= bottom:
                visible = getattr(self, "chart_series_visible", {})
                visible[label] = not visible.get(label, True)
                self.chart_series_visible = visible
                self._draw_chart()
                return

    def _legend_toggle(self, label: str) -> None:
        """Apply a legend checkbox change and redraw the chart."""
        variable = self.chart_legend_vars.get(label)
        if variable is not None:
            is_visible = bool(variable.get())
            self.chart_series_visible[label] = is_visible
            widget = self.chart_legend_widgets.get(label)
            if widget is not None:
                color = next((color for name, _values, color in self.chart_series if name == label), "#202124")
                widget.configure(fg=color if is_visible else "#9aa0a6")
        self._draw_chart()

    def _render_chart_legend(self) -> None:
        """Render a spacious, multi-column legend below the plotting area."""
        if not hasattr(self, "legend_frame"):
            return
        for child in self.legend_frame.winfo_children():
            child.destroy()
        self.chart_legend_vars = {}
        self.chart_legend_widgets = {}
        series = getattr(self, "chart_series", [])
        if not series:
            return
        for index, (label, _values, color) in enumerate(series):
            variable = tk.BooleanVar(value=self.chart_series_visible.get(label, True))
            self.chart_legend_vars[label] = variable
            check = tk.Checkbutton(
                self.legend_frame,
                text=label,
                variable=variable,
                command=lambda name=label: self._legend_toggle(name),
                anchor="w",
                justify="left",
                padx=4,
                pady=2,
                fg=color if variable.get() else "#9aa0a6",
                activeforeground=color,
                selectcolor="#ffffff",
                relief="flat",
                highlightthickness=0,
            )
            check.grid(row=index // 3, column=index % 3, sticky="ew", padx=(0, 12), pady=1)
            self.chart_legend_widgets[label] = check

    def _plot_changed(self) -> None:
        self._update_signal_options()
        self._update_3d_control()
        self._refresh_chart()

    def _group_changed(self) -> None:
        self._sync_plot_options()
        self._refresh_chart()

    def _sync_plot_options(self) -> None:
        groups = available_plot_groups(self.data)
        self.group_selector.configure(values=tuple(groups))
        if self.group_var.get() not in groups:
            self.group_var.set(next(iter(groups)))
        views = groups[self.group_var.get()]
        self.plot_selector.configure(values=views)
        if self.plot_var.get() not in views:
            self.plot_var.set(views[0])
        self._update_signal_options()
        self._update_3d_control()

    def _update_3d_control(self) -> None:
        if self.plot_var.get() == "3D trajectory":
            if not self.reset_3d_button.winfo_manager():
                self.reset_3d_button.pack(side="right")
        else:
            self.eef_drag_at = None
            self.reset_3d_button.pack_forget()

    def _reset_3d_view(self) -> None:
        self.eef_yaw, self.eef_pitch, self.eef_zoom = 0.0, 0.45, 1.0
        self._draw_chart()

    def _eef_drag_start(self, event: tk.Event) -> None:
        if self.plot_var.get() == "3D trajectory":
            self.eef_drag_at = (event.x, event.y)

    def _eef_drag_move(self, event: tk.Event) -> None:
        if self.plot_var.get() != "3D trajectory" or self.eef_drag_at is None:
            return
        last_x, last_y = self.eef_drag_at
        self.eef_yaw += (event.x - last_x) * 0.012
        self.eef_pitch = max(-1.35, min(1.35, self.eef_pitch + (event.y - last_y) * 0.012))
        self.eef_drag_at = (event.x, event.y)
        self._draw_chart()

    def _eef_drag_end(self, _event: tk.Event) -> None:
        self.eef_drag_at = None

    def _eef_mouse_wheel(self, event: tk.Event) -> None:
        if self.plot_var.get() != "3D trajectory":
            return
        upward = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
        self.eef_zoom = max(0.55, min(3.5, self.eef_zoom * (1.1 if upward else 0.9)))
        self._draw_chart()

    def _update_signal_options(self) -> None:
        if self.data is None:
            return
        plot = self.plot_var.get()
        if plot in MOTION_VIEWS:
            if self.data.kind == "episode":
                options = (
                    "Joint L2 norm",
                    *(name for name in self.data.names if "gripper" not in name.lower()),
                )
            else:
                options = ("Joint L2 norm", *motion_joint_names(self.data))
        elif plot in {"Position", "Tracking error"}:
            options = self.data.names or ("No joints",)
        elif plot == "3D trajectory":
            options = ("Both arms", "Left arm", "Right arm")
        else:
            options = ("—",)
        self.signal_selector.configure(values=options, state="readonly" if len(options) > 1 else "disabled")
        if self.signal_var.get() not in options:
            self.signal_var.set(options[0])

    def _make_series(self, start: int, end: int) -> tuple[list[tuple[str, np.ndarray, str]], np.ndarray, str]:
        assert self.data is not None
        data = self.data
        times = data.timestamps[start : end + 1] - data.timestamps[0]
        signal = self.signal_var.get()
        try:
            signal_index = data.names.index(signal)
        except ValueError:
            signal_index = None
        plot = self.plot_var.get()
        self.chart_markers = []
        self.chart_series_x = []
        self.chart_series_dash_patterns = []
        if plot in TRAJECTORY_VIEWS:
            model_switches, sent_switches = trajectory_chunk_switch_times(data, start, end)
            self.chart_markers = [
                (label, switches, color)
                for label, switches, color in (
                    ("New prediction", model_switches, "#1a73e8"),
                    ("Sent chunk switch", sent_switches, "#e76f51"),
                )
                if len(switches)
            ]
        if plot in MOTION_VIEWS:
            order, unit = MOTION_VIEWS[plot]
            if data.kind == "episode":
                joint_indices = tuple(
                    index for index, name in enumerate(data.names[: data.measured.shape[1]])
                    if "gripper" not in name.lower()
                )
                if not joint_indices:
                    joint_indices = tuple(range(data.measured.shape[1]))
                if signal == "Joint L2 norm":
                    selected_index = None
                else:
                    try:
                        selected_index = data.names.index(signal)
                    except ValueError:
                        selected_index = joint_indices[0]
                selected_times = times
                series = []
                for label, values, color in (
                    ("Measured", data.measured[start : end + 1], "#68707d"),
                    ("Recorded target", data.desired[start : end + 1], "#e76f51"),
                ):
                    x, derivatives = _episode_motion_series(
                        selected_times,
                        values,
                        order=order,
                        joint_indices=joint_indices,
                    )
                    if selected_index is None:
                        curve = np.linalg.norm(derivatives, axis=1)
                    else:
                        local_index = joint_indices.index(selected_index) if selected_index in joint_indices else None
                        if local_index is None:
                            curve = np.full(len(x), np.nan, dtype=np.float64)
                        else:
                            curve = derivatives[:, local_index]
                    series.append((label, curve, color))
                    self.chart_series_x.append(x)
                    self.chart_series_dash_patterns.append(None)
                return series, np.concatenate(self.chart_series_x), unit
            joints = motion_joint_names(data)
            joint_index = joints.index(signal) if signal in joints else None
            model_x, current_values, older_values, _ = policy_trajectory_series(
                data, start, end, order=order, joint_index=joint_index,
            )
            smooth_x, smooth_values, _, _ = policy_trajectory_series(
                data, start, end, order=order, joint_index=joint_index,
                smoothed=True,
            )
            sent_x, sent_values, _ = trajectory_motion_series(
                data, start, end, stream="command_sent", order=order, joint_index=joint_index,
            )
            series: list[tuple[str, np.ndarray, str]] = []
            if np.isfinite(older_values).any():
                series.append(("Earlier prediction", older_values, "#8ab4f8"))
                self.chart_series_x.append(model_x)
                self.chart_series_dash_patterns.append((4, 3))
            series.extend((
                ("Policy output", current_values, "#1a73e8"),
                ("Smoothed policy", smooth_values, "#0f9d8a"),
                ("Command sent", sent_values, "#e76f51"),
            ))
            self.chart_series_x.extend((model_x, smooth_x, sent_x))
            self.chart_series_dash_patterns.extend((None, None, None))
            return series, np.concatenate(self.chart_series_x), unit
        if plot in CHUNK_VIEWS:
            metric, unit = CHUNK_VIEWS[plot]
            x, model, sent = jitter_plot_series(data, start, end, metric)
            return [("Policy output", model, "#1a73e8"), ("Command sent", sent, "#e76f51")], x, unit
        if plot == "Inference latency":
            records = [r for r in data.command_records if isinstance(r.get("captured_at"), (int, float))]
            records = [r for r in records if times[0] + data.timestamps[0] <= float(r["captured_at"]) <= times[-1] + data.timestamps[0]]
            x = np.asarray([float(r["captured_at"]) - data.timestamps[0] for r in records])
            series = []
            for label, key, color in (("Round trip", "round_trip_ms", "#1a73e8"), ("Model", "model_inference_ms", "#e76f51"), ("Upload", "observation_upload_ms", "#0f9d8a")):
                values = [_timing_value_ms(r, key) for r in records]
                series.append((label, np.asarray(values), color))
            return series, x, "milliseconds"
        if plot == "Control interval":
            x = times[1:]
            return [("Control interval", np.diff(data.timestamps[start : end + 1]) * 1000.0, "#1a73e8")], x, "milliseconds"
        if plot == "3D trajectory":
            return [], times, "meters"
        desired = data.desired[start : end + 1]
        measured = data.measured[start : end + 1]
        index = 0 if signal_index is None else signal_index
        if plot == "Tracking error":
            self.chart_series_x = [times]
            self.chart_series_dash_patterns = [None]
            if data.kind == "deployment" and data.command_joints_rad.ndim == 2:
                # ``command_action`` is the queued model target and can be
                # many steps ahead of feedback.  Use the quantized joint
                # reference actually published to Piper for tracking error.
                measured_joints = _deployment_joint_values(measured)
                sent = data.command_joints_rad[start : end + 1]
                command_index = _deployment_joint_index(index, measured.shape[1])
                if command_index is not None and command_index < sent.shape[1] and command_index < measured_joints.shape[1]:
                    command_names = motion_joint_names(data)
                    return [(
                        f"{command_names[command_index]} command - measured",
                        sent[:, command_index] - measured_joints[:, command_index],
                        "#7c4dff",
                    )], times, "command - measured"
            return [(f"{data.names[index]} target - measured", desired[:, index] - measured[:, index], "#7c4dff")], times, "target - measured"
        series = [
            ("Measured", measured[:, index], "#68707d"),
            ("Recorded target", desired[:, index], "#e76f51"),
        ]
        if data.kind == "deployment" and data.command_joints_rad.ndim == 2:
            measured_joints = _deployment_joint_values(measured)
            sent = data.command_joints_rad[start : end + 1]
            command_index = _deployment_joint_index(index, measured.shape[1])
            if command_index is not None and command_index < sent.shape[1] and command_index < measured_joints.shape[1]:
                series.append(("Command sent", sent[:, command_index], "#9c27b0"))
        self.chart_series_x = [times for _ in series]
        self.chart_series_dash_patterns = [None for _ in series]
        model_joints = motion_joint_names(data)
        model_joint_index = model_joints.index(signal) if signal in model_joints else None
        if model_joint_index is None and signal_index is not None:
            state_width = data.measured.shape[1]
            if state_width == 7 and signal_index < 6:
                model_joint_index = signal_index
            elif state_width == 14:
                if signal_index < 6:
                    model_joint_index = signal_index
                elif 7 <= signal_index < 13:
                    model_joint_index = signal_index - 1
            elif state_width in {6, 12} and signal_index < len(model_joints):
                model_joint_index = signal_index
        if model_joint_index is not None:
            model_x, current, older, _ = policy_trajectory_series(
                data, start, end, order=0, joint_index=model_joint_index,
            )
            smooth_x, smooth_current, _, _ = policy_trajectory_series(
                data, start, end, order=0, joint_index=model_joint_index,
                smoothed=True,
            )
            if np.isfinite(older).any():
                series.append(("Earlier prediction", older, "#8ab4f8"))
                self.chart_series_x.append(model_x)
                self.chart_series_dash_patterns.append((4, 3))
            if np.isfinite(current).any():
                series.append(("Policy output", current, "#1a73e8"))
                self.chart_series_x.append(model_x)
                self.chart_series_dash_patterns.append(None)
            if np.isfinite(smooth_current).any():
                series.append(("Smoothed policy", smooth_current, "#0f9d8a"))
                self.chart_series_x.append(smooth_x)
                self.chart_series_dash_patterns.append(None)
        return series, np.concatenate(self.chart_series_x), "joint position (rad)"

    def _draw_chart(self) -> None:
        canvas = self.chart
        if canvas is None:
            return
        canvas.delete("all")
        self.chart_legend_regions = []
        width = max(320, canvas.winfo_width())
        height = max(220, canvas.winfo_height())
        if self.plot_var.get() == "3D trajectory" and self.data is not None:
            try:
                poses = self.pose_cache.get(self.data.path)
                if poses is None:
                    poses = compute_end_effector_positions(self.data)
                    self.pose_cache[self.data.path] = poses
                start, end = self._selected_indices()
                draw_end_effector_trajectory(
                    canvas, self.data, poses, start, end, self.signal_var.get(),
                    width=width, height=height, yaw=self.eef_yaw,
                    pitch=self.eef_pitch, zoom=self.eef_zoom, font_name=self.font_name,
                    chunk_switch_indices=sent_chunk_switch_indices(self.data, start, end),
                )
            except Exception as exc:
                canvas.create_text(width / 2, height / 2, text=f"3D trajectory unavailable: {exc}",
                                   fill="#68707d", font=(self.font_name, 10))
            return
        left, top, right, bottom = 58, 40 if self.chart_markers else 24, 18, 40
        if not self.chart_series or not len(self.chart_x):
            canvas.create_text(width // 2, height // 2, text="Select a source to display a chart", fill="#68707d", font=(self.font_name, 11))
            return
        visible = getattr(self, "chart_series_visible", {})
        visible_series = [
            (label, values, color)
            for label, values, color in self.chart_series
            if visible.get(label, True)
        ]
        all_values = np.concatenate([
            np.asarray(values, dtype=float).reshape(-1)
            for _, values, _ in (visible_series or self.chart_series)
        ])
        all_values = all_values[np.isfinite(all_values)]
        if not len(all_values):
            canvas.create_text(width // 2, height // 2, text="No valid values in this range", fill="#68707d", font=(self.font_name, 11))
            return
        ymin, ymax = float(np.min(all_values)), float(np.max(all_values))
        if math.isclose(ymin, ymax):
            pad = max(1.0, abs(ymin) * 0.1)
            ymin, ymax = ymin - pad, ymax + pad
        else:
            pad = (ymax - ymin) * 0.08
            ymin, ymax = ymin - pad, ymax + pad
        finite_x = self.chart_x[np.isfinite(self.chart_x)]
        if not len(finite_x):
            return
        xmin, xmax = float(np.min(finite_x)), float(np.max(finite_x))
        if math.isclose(xmin, xmax):
            xmax = xmin + 1.0
        xscale = (width - left - right) / (xmax - xmin)
        yscale = (height - top - bottom) / (ymax - ymin)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + fraction * (height - top - bottom)
            value = ymax - fraction * (ymax - ymin)
            canvas.create_line(left, y, width - right, y, fill="#edf0f4")
            canvas.create_text(left - 8, y, text=f"{value:.3g}", fill="#68707d", anchor="e", font=(self.font_name, 8))
        canvas.create_line(left, top, left, height - bottom, fill="#9aa0a6")
        canvas.create_line(left, height - bottom, width - right, height - bottom, fill="#9aa0a6")
        canvas.create_text(left, height - 14, text=f"{xmin:.2f}s", fill="#68707d", anchor="w", font=(self.font_name, 8))
        canvas.create_text(width - right, height - 14, text=f"{xmax:.2f}s", fill="#68707d", anchor="e", font=(self.font_name, 8))
        canvas.create_text(10, top, text=self.chart_y_label, fill="#68707d", anchor="nw", font=(self.font_name, 8))
        if not visible_series:
            canvas.create_text(width / 2, (top + height - bottom) / 2,
                               text="All curves are hidden", fill="#68707d",
                               font=(self.font_name, 11))
        dash_patterns = getattr(self, "chart_series_dash_patterns", [])
        if self.chart_markers:
            for _marker_label, marker_times, color in self.chart_markers:
                for marker in marker_times:
                    if xmin <= marker <= xmax:
                        x_pixel = left + (float(marker) - xmin) * xscale
                        canvas.create_line(x_pixel, top, x_pixel, height - bottom, fill=color, dash=(4, 3), width=1)
            for index, (marker_label, _times, color) in enumerate(self.chart_markers):
                marker_x = left + 8 + index * min(180, max(120, (width - left - right) // 2))
                canvas.create_line(marker_x, 26, marker_x + 14, 26, fill=color, dash=(4, 3))
                canvas.create_text(marker_x + 18, 26, text=marker_label, fill="#68707d", anchor="w", font=(self.font_name, 8))
        for series_index, (label, values, color) in enumerate(self.chart_series):
            is_visible = visible.get(label, True)
            if not is_visible:
                continue
            points = []
            values = np.asarray(values, dtype=float)
            series_x = (
                self.chart_series_x[series_index]
                if len(self.chart_series_x) == len(self.chart_series)
                else self.chart_x
            )
            dash_pattern = (
                dash_patterns[series_index]
                if series_index < len(dash_patterns) else None
            )
            limit = min(len(series_x), len(values))
            if self.plot_var.get() in CHUNK_VIEWS:
                for x_value, y_value in zip(series_x[:limit], values[:limit]):
                    if not np.isfinite(y_value):
                        continue
                    x_pixel = left + (float(x_value) - xmin) * xscale
                    y_pixel = top + (ymax - float(y_value)) * yscale
                    points.extend((x_pixel, y_pixel))
                    canvas.create_oval(x_pixel - 2, y_pixel - 2, x_pixel + 2, y_pixel + 2, fill=color, outline=color)
                if len(points) > 3:
                    canvas.create_line(*points, fill=color, width=1.5, smooth=False)
                continue
            for x_value, y_value in zip(series_x[:limit], values[:limit]):
                if not np.isfinite(y_value):
                    # Canvas lines need at least two (x, y) points. A one
                    # sample run can occur at NaN-separated forecast tails.
                    if len(points) > 3:
                        canvas.create_line(*points, fill=color, width=1.5, smooth=False,
                                           dash=dash_pattern or "")
                    points = []
                    continue
                points.extend((left + (float(x_value) - xmin) * xscale, top + (ymax - float(y_value)) * yscale))
            if len(points) > 3:
                canvas.create_line(*points, fill=color, width=1.5, smooth=False,
                                   dash=dash_pattern or "")
