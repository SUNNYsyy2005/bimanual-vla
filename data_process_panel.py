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

from data_process_analysis import (
    AnalysisData,
    compute_metrics,
    compute_end_effector_positions,
    load_analysis_data,
    scan_analysis_sources,
    selection_indices,
)


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
        self.plot_var = tk.StringVar(value="Measured position")
        self.signal_var = tk.StringVar(value="All dimensions")
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
                "rejected",
                "unsafe",
                "discarded",
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
        for column in range(5):
            summary.columnconfigure(column, weight=1)
        cards = (
            ("source", "Source"), ("duration", "Selected duration"),
            ("control", "Control rate"), ("inference", "Model commands"),
            ("latency", "Round-trip P50 / P95"), ("jitter", "Tick jitter P95"),
            ("sent", "Command coverage"), ("rejected", "Rejected actions"),
            ("unsafe", "Unsafe drops"), ("discarded", "Discarded total"),
        )
        for index, (key, title) in enumerate(cards):
            row, column = divmod(index, 5)
            card = tk.Frame(summary, bg="#f7f8fa", highlightthickness=1, highlightbackground="#e4e7ec")
            card.grid(row=row, column=column, sticky="ew", padx=4, pady=4)
            tk.Label(card, text=title, bg="#f7f8fa", fg="#68707d", font=("Liberation Serif", 9), anchor="w").pack(fill="x", padx=8, pady=(6, 1))
            tk.Label(card, textvariable=self.metric_vars[key], bg="#f7f8fa", fg="#202124", font=("Liberation Serif", 10, "bold"), anchor="w").pack(fill="x", padx=8, pady=(0, 6))

        accounting = ttk.LabelFrame(right, text="Action accounting", padding=8)
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
        ttk.Label(chart_controls, text="View").pack(side="left")
        plot_selector = ttk.Combobox(chart_controls, textvariable=self.plot_var, state="readonly", values=("Measured position", "Target vs measured", "Action velocity", "Action error", "Latency", "Tick interval", "End-effector XY", "End-effector XYZ"), width=23)
        plot_selector.pack(side="left", padx=(8, 14))
        plot_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_chart())
        ttk.Label(chart_controls, text="Signal").pack(side="left")
        self.signal_selector = ttk.Combobox(chart_controls, textvariable=self.signal_var, state="readonly", values=("All dimensions",), width=24)
        self.signal_selector.pack(side="left", padx=8)
        self.signal_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_chart())

        self.chart = tk.Canvas(right, background="#ffffff", highlightthickness=1, highlightbackground="#d9dde5")
        self.chart.grid(row=4, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())

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
        self.start_scale.configure(to=max(0.01, data.duration_s))
        self.end_scale.configure(to=max(0.01, data.duration_s))
        self.start_scale.set(0.0)
        self.end_scale.set(data.duration_s)
        self.start_var.set("0.00")
        self.end_var.set(f"{data.duration_s:.2f}")
        self.signal_selector.configure(values=("All dimensions", *data.names))
        self.signal_var.set("All dimensions")
        self.status_var.set(f"Loaded {data.label} · {data.sample_count} samples")
        self.apply_range()

    def _clear_view(self) -> None:
        for variable in self.metric_vars.values():
            variable.set("—")
        self.selection_var.set("No data selected")
        if self.chart is not None:
            self.chart.delete("all")
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)

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
        self.metric_vars["rejected"].set(str(rejected_rows))
        self.metric_vars["unsafe"].set(str(unsafe))
        self.metric_vars["discarded"].set(str(discarded))
        self.metric_vars["action_rows"].set(str(metrics.get("model_action_rows") or 0))
        self.metric_vars["executed_actions"].set(str(metrics.get("executed_control_actions") or 0))
        self.metric_vars["rejected_rows"].set(str(rejected_rows))
        self.metric_vars["unsafe_events"].set(str(unsafe))
        self.metric_vars["discarded_rows"].set(str(discarded))

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
            self._draw_chart()
            return
        start, end = self._selected_indices()
        if end < start:
            return
        self.chart_series, self.chart_x, self.chart_y_label = self._make_series(start, end)
        self._draw_chart()

    def _make_series(self, start: int, end: int) -> tuple[list[tuple[str, np.ndarray, str]], np.ndarray, str]:
        assert self.data is not None
        data = self.data
        times = data.timestamps[start : end + 1] - data.timestamps[0]
        signal = self.signal_var.get()
        try:
            signal_index = data.names.index(signal)
        except ValueError:
            signal_index = None
        colors = ("#1a73e8", "#e76f51", "#0f9d8a", "#7c4dff", "#f4b400", "#ab47bc", "#00acc1", "#5f6368")
        plot = self.plot_var.get()
        if plot == "Latency":
            records = [r for r in data.command_records if isinstance(r.get("captured_at"), (int, float))]
            records = [r for r in records if times[0] + data.timestamps[0] <= float(r["captured_at"]) <= times[-1] + data.timestamps[0]]
            x = np.asarray([float(r["captured_at"]) - data.timestamps[0] for r in records])
            series = []
            for label, key, color in (("Round trip", "round_trip_ms", "#1a73e8"), ("Model", "model_inference_ms", "#e76f51"), ("Upload", "observation_upload_ms", "#0f9d8a")):
                values = [float((r.get("_client_transport_timing") or {}).get(key, np.nan)) for r in records]
                series.append((label, np.asarray(values), color))
            return series, x, "milliseconds"
        if plot == "Tick interval":
            x = times[1:]
            return [("Tick interval", np.diff(data.timestamps[start : end + 1]) * 1000.0, "#1a73e8")], x, "milliseconds"
        if plot in {"End-effector XY", "End-effector XYZ"}:
            poses = self.pose_cache.get(data.path)
            if poses is None:
                poses = compute_end_effector_positions(data)
                self.pose_cache[data.path] = poses
            series = []
            if not poses:
                return [], times, "FK unavailable for this data format"
            pose_colors = ("#1a73e8", "#e76f51", "#0f9d8a", "#7c4dff", "#f4b400", "#ab47bc")
            axes = (0, 1) if plot == "End-effector XY" else (0, 1, 2)
            axis_names = ("X", "Y", "Z")
            for side_index, side in enumerate(("left", "right")):
                key = f"{side}_measured"
                if key not in poses:
                    continue
                values = poses[key][start : end + 1]
                for axis in axes:
                    series.append((f"{side} {axis_names[axis]}", values[:, axis], pose_colors[(side_index * 3 + axis) % len(pose_colors)]))
            return series, times, "meters"
        desired = data.desired[start : end + 1]
        measured = data.measured[start : end + 1]
        if plot == "Action velocity":
            values = np.diff(desired, axis=0)
            x = times[1:]
            y_label = "action delta / tick"
        elif plot == "Action error":
            values = desired - measured
            x = times
            y_label = "target - measured"
        else:
            values = measured
            x = times
            y_label = "joint value"
        series: list[tuple[str, np.ndarray, str]] = []
        if plot == "Target vs measured":
            index = 0 if signal_index is None else signal_index
            series = [(f"{data.names[index]} measured", measured[:, index], "#1a73e8"), (f"{data.names[index]} target", desired[:, index], "#e76f51")]
            return series, x, "joint / gripper value"
        if signal_index is not None:
            series.append((data.names[signal_index], values[:, signal_index], colors[signal_index % len(colors)]))
        else:
            for index, name in enumerate(data.names):
                series.append((name, values[:, index], colors[index % len(colors)]))
        return series, x, y_label

    def _draw_chart(self) -> None:
        canvas = self.chart
        if canvas is None:
            return
        canvas.delete("all")
        width = max(320, canvas.winfo_width())
        height = max(220, canvas.winfo_height())
        left, top, right, bottom = 58, 24, 18, 40
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
        xmin, xmax = float(self.chart_x[0]), float(self.chart_x[-1])
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
        legend_x = left + 8
        for label, values, color in self.chart_series:
            points = []
            values = np.asarray(values, dtype=float)
            limit = min(len(self.chart_x), len(values))
            for x_value, y_value in zip(self.chart_x[:limit], values[:limit]):
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
