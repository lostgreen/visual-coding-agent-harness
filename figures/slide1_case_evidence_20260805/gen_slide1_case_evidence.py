#!/usr/bin/env python3
"""Render slide-ready MM-Lifelong case-evidence figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = json.loads((ROOT / "evidence_data.json").read_text(encoding="utf-8"))

BG = "#F4F5F2"
PAPER = "#FFFFFF"
INK = "#202A2E"
MUTED = "#667176"
LINE = "#D9DEDD"
TEAL = "#177E78"
BLUE = "#3E6F9F"
GOLD = "#D5A028"
CORAL = "#C84F42"
PALE_TEAL = "#E7F1EF"
PALE_BLUE = "#EAF0F5"
PALE_GOLD = "#F7EFD8"
PALE_CORAL = "#F7E6E2"

FONT_LIGHT = Path("/System/Library/Fonts/STHeiti Light.ttc")
FONT_MEDIUM = Path("/System/Library/Fonts/STHeiti Medium.ttc")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "savefig.pad_inches": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


def font(size: float, *, bold: bool = False) -> FontProperties:
    return FontProperties(fname=str(FONT_MEDIUM if bold else FONT_LIGHT), size=size)


def canvas() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(6.4, 3.6), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 8,
    bold: bool = False,
    color: str = INK,
    ha: str = "left",
    va: str = "center",
    linespacing: float = 1.25,
    zorder: int = 5,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        color=color,
        fontproperties=font(size, bold=bold),
        linespacing=linespacing,
        transform=ax.transAxes,
        zorder=zorder,
    )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = PAPER,
    edge: str = "none",
    linewidth: float = 0.8,
    radius: float = 0.012,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def crop_to_aspect(
    image: Image.Image, aspect: float, focus: tuple[float, float]
) -> Image.Image:
    width, height = image.size
    current = width / height
    fx, fy = focus
    if current > aspect:
        new_width = int(height * aspect)
        left = int((width - new_width) * fx)
        left = max(0, min(left, width - new_width))
        return image.crop((left, 0, left + new_width, height))
    new_height = int(width / aspect)
    top = int((height - new_height) * fy)
    top = max(0, min(top, height - new_height))
    return image.crop((0, top, width, top + new_height))


def image_panel(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    image_path: str,
    *,
    focus: tuple[float, float] = (0.5, 0.5),
    border: str = PAPER,
    linewidth: float = 1.0,
) -> plt.Axes:
    x, y, w, h = rect
    physical_aspect = (w * 6.4) / (h * 3.6)
    image = Image.open(ROOT / image_path).convert("RGB")
    image = crop_to_aspect(image, physical_aspect, focus)
    image_ax = fig.add_axes(rect, zorder=2)
    image_ax.imshow(image)
    image_ax.axis("off")
    image_ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor=border,
            linewidth=linewidth,
            transform=image_ax.transAxes,
            clip_on=False,
        )
    )
    return image_ax


def image_badge(
    image_ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    face: str,
    color: str,
    size: float = 5.2,
    va: str = "top",
) -> None:
    image_ax.text(
        x,
        y,
        text,
        ha="left",
        va=va,
        color=color,
        fontproperties=font(size, bold=True),
        transform=image_ax.transAxes,
        zorder=8,
        bbox={
            "boxstyle": "round,pad=0.32,rounding_size=0.4",
            "facecolor": face,
            "edgecolor": "none",
            "alpha": 0.94,
        },
    )


def tag(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    face: str,
    color: str,
    width: float,
    height: float = 0.045,
    size: float = 5.8,
) -> None:
    box(ax, x, y - height / 2, width, height, face=face, radius=0.008, zorder=6)
    add_text(
        ax,
        x + width / 2,
        y,
        text,
        size=size,
        bold=True,
        color=color,
        ha="center",
        zorder=7,
    )


def header(ax: plt.Axes, case: str, title: str, takeaway: str, accent: str) -> None:
    add_text(ax, 0.055, 0.945, case, size=5.7, bold=True, color=accent)
    add_text(ax, 0.945, 0.945, takeaway, size=5.7, bold=True, color=accent, ha="right")
    add_text(ax, 0.055, 0.875, title, size=14.5, bold=True)
    ax.add_patch(
        Rectangle(
            (0.055, 0.825), 0.075, 0.009, facecolor=accent, transform=ax.transAxes
        )
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches=None
    )
    fig.savefig(OUT / f"{stem}.pdf", facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


def render_case_0031() -> None:
    d = DATA["case_0031"]
    fig, ax = canvas()
    header(
        ax,
        "CASE 0031 · LANGUAGE CONTENT RECALL",
        "同一 material，三次读取，三种唱词",
        "TOOL CALL ≠ NEW EVIDENCE",
        TEAL,
    )

    image_ax = image_panel(
        fig, (0.055, 0.255, 0.44, 0.505), d["frame"], focus=(0.5, 0.58)
    )
    image_ax.add_patch(
        Rectangle(
            (0.32, 0.045),
            0.39,
            0.085,
            fill=False,
            edgecolor=GOLD,
            linewidth=1.8,
            transform=image_ax.transAxes,
        )
    )
    image_badge(
        image_ax, 0.025, 0.965, "SAME VISUAL WINDOW", face=PALE_GOLD, color="#8B6410"
    )
    image_badge(
        image_ax,
        0.025,
        0.035,
        "[18011, 18047] · 72 frames · 2 fps",
        face="#202A2E",
        color=PAPER,
        va="bottom",
    )

    box(ax, 0.525, 0.255, 0.42, 0.505, face=PAPER, edge=LINE, radius=0.008)
    add_text(ax, 0.548, 0.723, "Saved visual outputs", size=6.0, bold=True, color=MUTED)
    row_y = [0.635, 0.505, 0.375]
    fills = [PALE_BLUE, PALE_TEAL, PALE_CORAL]
    accents = [BLUE, TEAL, CORAL]
    for reading, y, fill, accent in zip(d["readings"], row_y, fills, accents):
        box(ax, 0.545, y - 0.052, 0.38, 0.102, face=fill, radius=0.006, zorder=2)
        ax.add_patch(
            Rectangle(
                (0.545, y - 0.052),
                0.006,
                0.102,
                facecolor=accent,
                transform=ax.transAxes,
                zorder=3,
            )
        )
        add_text(
            ax, 0.563, y + 0.028, reading["run"], size=5.1, bold=True, color=accent
        )
        add_text(ax, 0.563, y - 0.014, f"“{reading['text']}”", size=6.35, bold=True)

    box(ax, 0.055, 0.075, 0.89, 0.105, face=PALE_TEAL, radius=0.008)
    add_text(ax, 0.078, 0.128, "1 WINDOW", size=7.2, bold=True, color=TEAL)
    add_text(ax, 0.201, 0.128, "→", size=9, bold=True, color=MUTED, ha="center")
    add_text(ax, 0.285, 0.128, "3 READS", size=7.2, bold=True, color=TEAL, ha="center")
    add_text(ax, 0.391, 0.128, "→", size=9, bold=True, color=MUTED, ha="center")
    add_text(
        ax,
        0.515,
        0.128,
        "3 INCOMPATIBLE STRINGS",
        size=7.2,
        bold=True,
        color=CORAL,
        ha="center",
    )
    add_text(
        ax,
        0.925,
        0.128,
        "更多调用增加了 interpretation，\n没有自动增加稳定事实。",
        size=6.2,
        bold=True,
        color=INK,
        ha="right",
    )
    save(fig, "fig_case0031_tool_call_not_evidence")


def render_case_0038() -> None:
    d = DATA["case_0038"]
    fig, ax = canvas()
    header(
        ax,
        "CASE 0038 · EVENT TRACKING",
        "验证了一条证据链，但它来自错误 occurrence",
        "REFERENCE VALID ≠ GROUNDING",
        CORAL,
    )

    wrong_ax = image_panel(
        fig, (0.055, 0.315, 0.37, 0.42), d["wrong_frame"], focus=(0.52, 0.60)
    )
    gold_ax = image_panel(
        fig, (0.575, 0.315, 0.37, 0.42), d["gold_frame"], focus=(0.60, 0.48)
    )
    wrong_ax.add_patch(
        Rectangle(
            (0.34, 0.095),
            0.33,
            0.095,
            fill=False,
            edgecolor=CORAL,
            linewidth=1.8,
            transform=wrong_ax.transAxes,
        )
    )
    gold_ax.add_patch(
        Rectangle(
            (0.75, 0.57),
            0.245,
            0.25,
            fill=False,
            edgecolor=TEAL,
            linewidth=1.8,
            transform=gold_ax.transAxes,
        )
    )
    image_badge(
        wrong_ax, 0.025, 0.965, "INSPECTED 17830–17850", face=PALE_CORAL, color=CORAL
    )
    image_badge(
        gold_ax, 0.025, 0.965, "OFFICIAL CLUE 19950–19952", face=PALE_TEAL, color=TEAL
    )

    ax.add_patch(
        FancyArrowPatch(
            (0.455, 0.535),
            (0.545, 0.535),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=CORAL,
            transform=ax.transAxes,
            zorder=6,
        )
    )
    add_text(ax, 0.50, 0.585, "wrong", size=5.4, bold=True, color=CORAL, ha="center")
    add_text(
        ax, 0.50, 0.555, "occurrence", size=5.4, bold=True, color=CORAL, ha="center"
    )

    box(ax, 0.055, 0.19, 0.37, 0.092, face=PALE_CORAL, radius=0.006)
    add_text(ax, 0.073, 0.247, "画面 Boss：虎先锋", size=6.2, bold=True, color=CORAL)
    add_text(
        ax, 0.073, 0.211, "VLM：虎之锋  ·  reference_valid = true", size=5.8, bold=True
    )
    box(ax, 0.575, 0.19, 0.37, 0.092, face=PALE_TEAL, radius=0.006)
    add_text(ax, 0.593, 0.247, "官方 clue：虎伥掉落", size=6.2, bold=True, color=TEAL)
    add_text(
        ax, 0.593, 0.211, "画面 UI：旧拨浪鼓  ·  never retrieved", size=5.8, bold=True
    )

    box(ax, 0.055, 0.072, 0.89, 0.082, face=PALE_GOLD, radius=0.008)
    add_text(
        ax,
        0.078,
        0.113,
        "正确 occurrence 没进入 candidate set；后续 reference integrity 只能验证“引用是否存在”，不能验证“引用是否答对了题”。",
        size=6.45,
        bold=True,
    )
    save(fig, "fig_case0038_reference_not_grounding")


def render_case_0146() -> None:
    d = DATA["case_0146"]
    fig, ax = canvas()
    header(
        ax,
        "CASE 0146 · ATTRIBUTE RECOGNITION",
        "稀疏观察生成精确坐标，refinement 放大错误 cue",
        "EVIDENCE MACHINERY AMPLIFIES FALSE CUES",
        GOLD,
    )

    x0, x1, y = 0.08, 0.92, 0.735
    ax.plot([x0, x1], [y, y], color=LINE, linewidth=2, transform=ax.transAxes, zorder=1)
    for position in np.linspace(x0, x1, d["frames"]):
        ax.plot(
            [position, position],
            [y - 0.012, y + 0.012],
            color="#AEB8B6",
            linewidth=0.35,
            transform=ax.transAxes,
        )
    start = d["wide_window"][0]
    duration = d["duration_sec"]
    for claim in d["claims"]:
        position = x0 + (claim["virtual_time"] - start) / duration * (x1 - x0)
        ax.plot(
            position,
            y,
            marker="o",
            markersize=5.5,
            color=CORAL,
            transform=ax.transAxes,
            zorder=5,
        )
        add_text(
            ax,
            position,
            y + 0.045,
            f"{claim['label']} @{claim['virtual_time']:.3f}",
            size=5.1,
            bold=True,
            color=CORAL,
            ha="center",
        )
    add_text(ax, x0, y - 0.05, "72521.375", size=4.8, color=MUTED, ha="center")
    add_text(ax, x1, y - 0.05, "74563.475", size=4.8, color=MUTED, ha="center")
    tag(ax, 0.075, 0.805, "2042 sec", face=PALE_BLUE, color=BLUE, width=0.095)
    tag(ax, 0.18, 0.805, "96 frames", face=PALE_TEAL, color=TEAL, width=0.105)
    tag(
        ax,
        0.295,
        0.805,
        "actual 0.047 fps",
        face=PALE_GOLD,
        color="#8B6410",
        width=0.14,
    )
    add_text(
        ax,
        0.92,
        0.805,
        "≈ 1 sampled frame / 21.5 sec",
        size=5.5,
        bold=True,
        color=MUTED,
        ha="right",
    )

    left = d["claims"][0]
    right = d["claims"][1]
    burn_ax = image_panel(
        fig, (0.055, 0.245, 0.42, 0.40), left["frame"], focus=(0.50, 0.48)
    )
    freeze_ax = image_panel(
        fig, (0.525, 0.245, 0.42, 0.40), right["frame"], focus=(0.50, 0.48)
    )
    image_badge(burn_ax, 0.025, 0.96, "VLM CLAIM: BURN", face=PALE_CORAL, color=CORAL)
    image_badge(
        freeze_ax, 0.025, 0.96, "VLM CLAIM: FREEZE", face=PALE_CORAL, color=CORAL
    )
    image_badge(
        burn_ax,
        0.025,
        0.04,
        "实际：兵器铸造菜单",
        face="#202A2E",
        color=PAPER,
        va="bottom",
    )
    image_badge(
        freeze_ax,
        0.025,
        0.04,
        "实际：人物剧情画面",
        face="#202A2E",
        color=PAPER,
        va="bottom",
    )

    box(ax, 0.055, 0.073, 0.89, 0.105, face=PALE_GOLD, radius=0.008)
    add_text(ax, 0.078, 0.126, "2 FALSE CUES", size=6.8, bold=True, color=CORAL)
    add_text(ax, 0.224, 0.126, "→", size=8, bold=True, color=MUTED, ha="center")
    add_text(
        ax,
        0.328,
        0.126,
        "2 NARROW PROBES",
        size=6.8,
        bold=True,
        color=GOLD,
        ha="center",
    )
    add_text(ax, 0.462, 0.126, "→", size=8, bold=True, color=MUTED, ha="center")
    add_text(ax, 0.57, 0.126, "+8 FRAMES", size=6.8, bold=True, color=GOLD, ha="center")
    add_text(ax, 0.69, 0.126, "→", size=8, bold=True, color=MUTED, ha="center")
    add_text(
        ax,
        0.92,
        0.126,
        "candidate_only / invalid",
        size=6.6,
        bold=True,
        color=CORAL,
        ha="right",
    )
    save(fig, "fig_case0146_false_precision")


def render_phase_4() -> None:
    d = DATA["phase_4"]
    fig, ax = canvas()
    header(
        ax,
        "PHASE 4 · PAIRED GATE",
        "Runtime 接管 evidence state：看得更少，也没有更 grounded",
        "LESS LOOKING ≠ BETTER GROUNDING",
        BLUE,
    )

    labels = list(d["mean_frames"].keys())
    values = [d["mean_frames"][label] for label in labels]
    colors = ["#A9B2B4", TEAL, GOLD]
    bar_ax = fig.add_axes([0.145, 0.245, 0.45, 0.49], facecolor=BG)
    y = np.arange(len(labels))
    bars = bar_ax.barh(
        y, values, color=colors, height=0.48, edgecolor=PAPER, linewidth=0.8
    )
    bar_ax.set_xlim(0, 70)
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels(labels, fontproperties=font(6.2, bold=True), color=INK)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("Mean inspected frames", fontproperties=font(5.6), color=MUTED)
    bar_ax.grid(axis="x", color=LINE, linewidth=0.6)
    bar_ax.set_axisbelow(True)
    for spine in bar_ax.spines.values():
        spine.set_visible(False)
    bar_ax.tick_params(axis="x", colors=MUTED, labelsize=5)
    bar_ax.tick_params(axis="y", length=0)
    for idx, (bar, value) in enumerate(zip(bars, values)):
        bar_ax.text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            color=INK,
            fontproperties=font(6.2, bold=True),
        )
        if idx > 0:
            reduction = (1 - value / values[0]) * 100
            bar_ax.text(
                value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"−{reduction:.0f}%",
                ha="center",
                va="center",
                color=PAPER,
                fontproperties=font(5.7, bold=True),
            )

    box(ax, 0.645, 0.405, 0.30, 0.33, face=PAPER, edge=LINE, radius=0.008)
    add_text(
        ax,
        0.795,
        0.655,
        "GROUNDED CORRECT",
        size=5.8,
        bold=True,
        color=MUTED,
        ha="center",
    )
    add_text(ax, 0.795, 0.555, "0 / 10", size=22, bold=True, color=CORAL, ha="center")
    add_text(
        ax,
        0.795,
        0.467,
        "in both runtime-derived roots",
        size=5.5,
        bold=True,
        color=CORAL,
        ha="center",
    )

    box(ax, 0.645, 0.245, 0.30, 0.125, face=PALE_CORAL, radius=0.008)
    add_text(ax, 0.665, 0.325, "CASE 0117", size=5.5, bold=True, color=CORAL)
    add_text(
        ax,
        0.665,
        0.278,
        "Phase 3: correct + ref-valid @32f\nPhase 4: regressed in both roots",
        size=5.7,
        bold=True,
    )

    box(ax, 0.055, 0.072, 0.89, 0.092, face=PALE_BLUE, radius=0.008)
    add_text(
        ax,
        0.078,
        0.118,
        "Evidence machinery 进入 controller loop 后，观察成本下降，但 grounding 没有改善；减少观察本身不是成功信号。",
        size=6.5,
        bold=True,
    )
    save(fig, "fig_phase4_observation_collapse")


def render_overview() -> None:
    c31 = DATA["case_0031"]
    c38 = DATA["case_0038"]
    c146 = DATA["case_0146"]
    fig, ax = canvas()

    add_text(ax, 0.05, 0.94, "THREE CASES + ONE GATE", size=5.6, bold=True, color=TEAL)
    add_text(
        ax,
        0.05,
        0.875,
        "连续 NO-GO 的证据：问题发生在 Evidence Lifecycle，而不是单一 score",
        size=13.0,
        bold=True,
    )
    ax.add_patch(
        Rectangle((0.05, 0.825), 0.075, 0.009, facecolor=TEAL, transform=ax.transAxes)
    )

    panel_x = [0.05, 0.355, 0.66]
    panel_w = 0.285
    for x in panel_x:
        box(ax, x, 0.35, panel_w, 0.42, face=PAPER, edge=LINE, radius=0.008)

    add_text(ax, 0.07, 0.735, "0031", size=6.2, bold=True, color=TEAL)
    add_text(
        ax, 0.115, 0.735, "Tool call ≠ new evidence", size=5.4, bold=True, color=TEAL
    )
    image_panel(fig, (0.07, 0.515, 0.245, 0.18), c31["frame"], focus=(0.5, 0.58))
    add_text(ax, 0.07, 0.47, "同一 36 秒窗口", size=5.8, bold=True, color=MUTED)
    add_text(ax, 0.07, 0.415, "3 次读取 → 3 种唱词", size=7.0, bold=True, color=CORAL)
    add_text(
        ax, 0.07, 0.375, "interpretation 数量增加，事实未稳定", size=5.2, color=INK
    )

    add_text(ax, 0.375, 0.735, "0038", size=6.2, bold=True, color=CORAL)
    add_text(
        ax,
        0.42,
        0.735,
        "Reference valid ≠ grounding",
        size=5.25,
        bold=True,
        color=CORAL,
    )
    image_panel(fig, (0.375, 0.515, 0.115, 0.18), c38["wrong_frame"], focus=(0.5, 0.6))
    image_panel(fig, (0.505, 0.515, 0.115, 0.18), c38["gold_frame"], focus=(0.72, 0.45))
    add_text(ax, 0.432, 0.49, "wrong", size=4.8, bold=True, color=CORAL, ha="center")
    add_text(
        ax, 0.562, 0.49, "official clue", size=4.8, bold=True, color=TEAL, ha="center"
    )
    add_text(
        ax, 0.375, 0.425, "虎先锋证据链被判 valid", size=6.8, bold=True, color=CORAL
    )
    add_text(
        ax, 0.375, 0.375, "旧拨浪鼓 occurrence 从未进入 candidates", size=5.1, color=INK
    )

    add_text(ax, 0.68, 0.735, "0146", size=6.2, bold=True, color=GOLD)
    add_text(
        ax,
        0.725,
        0.735,
        "False cue amplification",
        size=5.4,
        bold=True,
        color="#8B6410",
    )
    image_panel(
        fig, (0.68, 0.515, 0.115, 0.18), c146["claims"][0]["frame"], focus=(0.5, 0.48)
    )
    image_panel(
        fig, (0.81, 0.515, 0.115, 0.18), c146["claims"][1]["frame"], focus=(0.5, 0.48)
    )
    add_text(ax, 0.737, 0.49, "“Burn”", size=4.8, bold=True, color=CORAL, ha="center")
    add_text(ax, 0.867, 0.49, "“Freeze”", size=4.8, bold=True, color=CORAL, ha="center")
    add_text(
        ax,
        0.68,
        0.425,
        "2 false cues → 8 more frames",
        size=6.8,
        bold=True,
        color=CORAL,
    )
    add_text(ax, 0.68, 0.375, "实际是菜单与剧情画面", size=5.2, color=INK)

    box(ax, 0.05, 0.07, 0.895, 0.22, face=PALE_BLUE, radius=0.008)
    add_text(ax, 0.073, 0.245, "PHASE 4 GATE", size=5.6, bold=True, color=BLUE)
    add_text(ax, 0.073, 0.182, "59.9", size=13, bold=True, color=MUTED)
    add_text(ax, 0.155, 0.182, "→", size=10, bold=True, color=MUTED, ha="center")
    add_text(
        ax, 0.26, 0.182, "19.9 / 36.4", size=13, bold=True, color=TEAL, ha="center"
    )
    add_text(ax, 0.073, 0.125, "mean frames", size=5.0, color=MUTED)
    add_text(ax, 0.50, 0.185, "但", size=7.0, bold=True, color=MUTED, ha="center")
    add_text(ax, 0.60, 0.182, "0 / 10", size=16, bold=True, color=CORAL, ha="center")
    add_text(
        ax,
        0.60,
        0.122,
        "Grounded Correct",
        size=5.2,
        bold=True,
        color=CORAL,
        ha="center",
    )
    add_text(ax, 0.765, 0.198, "0117", size=6.0, bold=True, color=CORAL)
    add_text(
        ax, 0.765, 0.145, "correct control\n→ regressed twice", size=6.0, bold=True
    )
    save(fig, "fig_slide1_case_evidence_overview")


def write_manifest() -> None:
    manifest: dict[str, object] = {"renderer": Path(__file__).name, "outputs": []}
    outputs = manifest["outputs"]
    assert isinstance(outputs, list)
    for path in sorted(OUT.glob("fig_*.*")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        outputs.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": digest}
        )
    (OUT / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    render_case_0031()
    render_case_0038()
    render_case_0146()
    render_phase_4()
    render_overview()
    write_manifest()
    print(f"Rendered 5 PNG/PDF figure pairs under {OUT}")


if __name__ == "__main__":
    main()
