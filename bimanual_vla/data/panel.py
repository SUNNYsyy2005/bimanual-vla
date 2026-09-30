"""Tk read-only Data process page for collected robot data."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
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
    trajectory_motion_series,
)
from bimanual_vla.data.eef_view import draw_end_effector_trajectory


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


def available_plot_groups(data: AnalysisData | None) -> dict[str, tuple[str, ...]]:
    """Only offer charts backed by the selected source's recording format."""
    if data is not None and data.kind == "episode":
        return {
            "Trajectory": ("Position", "Tracking error"),
            "Timing": ("Control interval",),
            "End effector": PLOT_GROUPS["End effector"],
        }
    return PLOT_GROUPS


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
        self.eef_yaw = 0.0
        self.eef_pitch = 0.45
        self.eef_zoom = 1.0
        self.eef_drag_at: tuple[int, int] | None = None
        self._build_ui()
        self.refresh_sources()

    def _build_ui(self) -> None:
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
        self.source_tree = ttk.Treeview(source_box, columns=("kind", "source"), show="headings", height=18)
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

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)
        summary = ttk.LabelFrame(right, text="Summary", padding=10)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 10))
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

        accounting = ttk.LabelFrame(right, text="Actions and chunk metrics", padding=8)
        accounting.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for column in range(4):
            accounting.columnconfigure(column, weight=1)
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
            card.grid(row=row, column=column, sticky="ew", padx=4, pady=3)
            tk.Label(card, text=title, bg="#f7f8fa", fg="#68707d", font=("Liberation Serif", 9), anchor="w").pack(fill="x", padx=8, pady=(5, 1))
            tk.Label(card, textvariable=self.metric_vars[key], bg="#f7f8fa", fg="#202124", font=("Liberation Serif", 10, "bold"), anchor="w").pack(fill="x", padx=8, pady=(0, 5))
        ttk.Label(
            accounting,
            text="Trajectory jitter · policy output uses the full horizon; sent uses issued joint commands",
            foreground="#68707d",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(5, 2))
        self.jitter_tree = ttk.Treeview(
            accounting, columns=("metric", "model", "sent"), show="headings", height=3
        )
        for key, title, width in (
            ("metric", "Metric", 240),
            ("model", "Policy output", 170),
            ("sent", "Command sent", 170),
        ):
            self.jitter_tree.heading(key, text=title)
            self.jitter_tree.column(key, width=width, minwidth=110, stretch=True)
        self.jitter_tree.grid(row=3, column=0, columnspan=4, sticky="ew", padx=4, pady=(0, 4))

        controls = ttk.LabelFrame(right, text="Analysis range", padding=8)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 10))
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

        chart_controls = ttk.Frame(right)
        chart_controls.grid(row=3, column=0, sticky="ew", pady=(0, 6))
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

        self.chart = tk.Canvas(right, background="#ffffff", highlightthickness=1, highlightbackground="#d9dde5")
        self.chart.grid(row=4, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())
        self.chart.bind("<ButtonPress-1>", self._eef_drag_start)
        self.chart.bind("<B1-Motion>", self._eef_drag_move)
        self.chart.bind("<ButtonRelease-1>", self._eef_drag_end)
        self.chart.bind("<MouseWheel>", self._eef_mouse_wheel)
        self.chart.bind("<Button-4>", self._eef_mouse_wheel)
        self.chart.bind("<Button-5>", self._eef_mouse_wheel)

        events = ttk.LabelFrame(right, text="Events in selected range", padding=6)
        events.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        events.columnconfigure(0, weight=1)
        self.events_tree = ttk.Treeview(events, columns=("event", "count"), show="headings", height=4)
        self.events_tree.heading("event", text="Event")
        self.events_tree.heading("count", text="Count")
        self.events_tree.column("event", width=520, anchor="w")
        self.events_tree.column("count", width=80, anchor="center", stretch=False)
        self.events_tree.grid(row=0, column=0, sticky="ew")

    def refresh_sources(self) -> None:
        current = self._selected_path()
        try:
            self.sources = scan_analysis_sources(self.roots_provider())
        except Exception as exc:  # keep refresh non-fatal
            self.sources = []
            self.status_var.set(f"Unable to scan data: {exc}")
        self._render_source_tree(current)

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
        try:
            data = load_analysis_data(path)
        except Exception as exc:
            self.data = None
            self.status_var.set(f"Cannot load {path.name}: {exc}")
            self._clear_view()
            return
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

    def _clear_view(self) -> None:
        for variable in self.metric_vars.values():
            variable.set("—")
        self.selection_var.set("No data selected")
        self.chart_series = []
        self.chart_x = np.array([], dtype=np.float64)
        self.chart_markers = []
        self.chart_series_x = []
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
            self._draw_chart()
            return
        self.chart_series, self.chart_x, self.chart_y_label = self._make_series(start, end)
        self._draw_chart()

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
        if plot in MOTION_VIEWS:
            order, unit = MOTION_VIEWS[plot]
            joints = motion_joint_names(data)
            joint_index = joints.index(signal) if signal in joints else None
            model_x, model_values, model_switches = trajectory_motion_series(
                data, start, end, stream="model_raw", order=order, joint_index=joint_index,
            )
            sent_x, sent_values, sent_switches = trajectory_motion_series(
                data, start, end, stream="command_sent", order=order, joint_index=joint_index,
            )
            self.chart_series_x = [model_x, sent_x]
            self.chart_markers = [
                ("New prediction", model_switches, "#1a73e8"),
                ("Sent chunk switch", sent_switches, "#e76f51"),
            ]
            return [
                ("Policy output", model_values, "#1a73e8"),
                ("Command sent", sent_values, "#e76f51"),
            ], np.concatenate((model_x, sent_x)), unit
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
            return [(f"{data.names[index]} target - measured", desired[:, index] - measured[:, index], "#7c4dff")], times, "target - measured"
        return [
            ("Measured", measured[:, index], "#1a73e8"),
            ("Recorded target", desired[:, index], "#e76f51"),
        ], times, "joint / gripper value"

    def _draw_chart(self) -> None:
        canvas = self.chart
        if canvas is None:
            return
        canvas.delete("all")
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
                )
            except Exception as exc:
                canvas.create_text(width / 2, height / 2, text=f"3D trajectory unavailable: {exc}",
                                   fill="#68707d", font=(self.font_name, 10))
            return
        motion_view = self.plot_var.get() in MOTION_VIEWS
        left, top, right, bottom = 58, 40 if motion_view else 24, 18, 40
        if not self.chart_series or not len(self.chart_x):
            canvas.create_text(width // 2, height // 2, text="Select a source to display a chart", fill="#68707d", font=(self.font_name, 11))
            return
        all_values = np.concatenate([np.asarray(values, dtype=float).reshape(-1) for _, values, _ in self.chart_series])
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
        if motion_view:
            for _marker_label, marker_times, color in self.chart_markers:
                for marker in marker_times:
                    if xmin <= marker <= xmax:
                        x_pixel = left + (float(marker) - xmin) * xscale
                        canvas.create_line(x_pixel, top, x_pixel, height - bottom, fill=color, dash=(4, 3), width=1)
            for index, (marker_label, _times, color) in enumerate(self.chart_markers):
                marker_x = left + 8 + index * min(180, max(120, (width - left - right) // 2))
                canvas.create_line(marker_x, 26, marker_x + 14, 26, fill=color, dash=(4, 3))
                canvas.create_text(marker_x + 18, 26, text=marker_label, fill="#68707d", anchor="w", font=(self.font_name, 8))
        legend_x = left + 8
        for series_index, (label, values, color) in enumerate(self.chart_series):
            points = []
            values = np.asarray(values, dtype=float)
            series_x = self.chart_series_x[series_index] if motion_view else self.chart_x
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
                canvas.create_line(legend_x, 10, legend_x + 14, 10, fill=color, width=2)
                canvas.create_text(legend_x + 18, 10, text=label, fill="#68707d", anchor="w", font=(self.font_name, 8))
                legend_x += min(145, max(70, len(label) * 7 + 30))
                continue
            for x_value, y_value in zip(series_x[:limit], values[:limit]):
                if not np.isfinite(y_value):
                    if len(points) > 1:
                        canvas.create_line(*points, fill=color, width=1.5, smooth=False)
                    points = []
                    continue
                points.extend((left + (float(x_value) - xmin) * xscale, top + (ymax - float(y_value)) * yscale))
            if len(points) > 3:
                canvas.create_line(*points, fill=color, width=1.5, smooth=False)
            canvas.create_line(legend_x, 10, legend_x + 14, 10, fill=color, width=2)
            canvas.create_text(legend_x + 18, 10, text=label, fill="#68707d", anchor="w", font=(self.font_name, 8))
            legend_x += min(145, max(70, len(label) * 7 + 30))
