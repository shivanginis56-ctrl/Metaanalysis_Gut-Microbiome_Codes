"""
generate_venn.py
One figure, two panels:
  Panel A — AE-119 (95% mass): 5 Venns (African, Asian, Caucasian, Hispanic, Overall)
  Panel B — AE-70  (Score>mean): 5 Venns
Style: large labelled regions, metrics above each Venn.
"""
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib_venn import venn2, venn2_circles
from pathlib import Path
import shutil

ROOT     = Path(__file__).resolve().parent.parent
RACES    = ["African", "Asian", "Caucasian", "Hispanic"]
AE_COLS  = [2, 3, 4, 5]
MAN_COLS = [6, 7, 8, 9]

COLOR_AE_119  = "#6BAED6"   # medium blue
COLOR_AE_70   = "#74C476"   # medium green
COLOR_MAN     = "#FEC44F"   # golden wheat
COLOR_SHARED_119 = "#B07FBF"  # purple — visible overlap for blue+wheat
COLOR_TEXT    = "#222222"


def load_data(xlsx_path):
    wb   = openpyxl.load_workbook(xlsx_path, data_only=True)
    data = {}
    for shname in ["AE_Core_119", "AE_Core_70"]:
        ws   = wb[shname]
        rows = list(ws.iter_rows(values_only=True))[2:]
        ae_pr  = {r: set() for r in RACES}
        man_pr = {r: set() for r in RACES}
        for row in rows:
            otu = row[0]
            if otu is None:
                continue
            for i, race in enumerate(RACES):
                if row[AE_COLS[i]]  == "✓": ae_pr[race].add(otu)
                if row[MAN_COLS[i]] == "✓": man_pr[race].add(otu)
        data[shname] = {"ae": ae_pr, "man": man_pr}
    return data


def draw_venn(ax, ae_set, man_set, race_title, ae_color, shared_color=None):
    only_ae  = ae_set  - man_set
    only_man = man_set - ae_set
    shared   = ae_set  & man_set

    p  = len(shared) / len(ae_set)  if ae_set  else 0
    r  = len(shared) / len(man_set) if man_set else 0
    f1 = 2 * p * r / (p + r)        if (p + r) else 0

    v = venn2(
        subsets=(len(only_ae), len(only_man), len(shared)),
        set_labels=("", ""),
        ax=ax,
        set_colors=(ae_color, COLOR_MAN),
        alpha=0.65,
    )
    venn2_circles(
        subsets=(len(only_ae), len(only_man), len(shared)),
        ax=ax, linewidth=0.8, color="#999999",
    )

    # Override shared region colour if specified
    if shared_color:
        patch = v.get_patch_by_id("11")
        if patch:
            patch.set_facecolor(shared_color)
            patch.set_alpha(0.80)

    # Label each region: name + count
    for pid, name, count in [
        ("10", "AE-only",     len(only_ae)),
        ("11", "Shared",      len(shared)),
        ("01", "Manual-only", len(only_man)),
    ]:
        lbl = v.get_label_by_id(pid)
        if lbl:
            lbl.set_text(f"{name}\n{count}")
            lbl.set_fontsize(9.5)
            lbl.set_fontweight("bold")
            lbl.set_color(COLOR_TEXT)

    # Tighten axis limits so circles fill the cell
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.62, 0.62)

    ax.set_title(
        f"{race_title}\n"
        f"P={p:.2f} | R={r:.2f} | F1={f1:.2f}\n"
        f"AE={len(ae_set)} | Manual={len(man_set)}",
        fontsize=9.5, fontweight="bold", pad=5, linespacing=1.5,
    )


def draw_row(fig, gs, row_idx, ae_per_race, man_per_race, ae_color,
             panel_letter, panel_title, shared_color=None):
    cols = RACES + ["Overall unique taxa"]
    axes = [fig.add_subplot(gs[row_idx, c]) for c in range(5)]

    for i, race in enumerate(cols):
        if race == "Overall unique taxa":
            ae_set  = set().union(*ae_per_race.values())
            man_set = set().union(*man_per_race.values())
        else:
            ae_set  = ae_per_race[race]
            man_set = man_per_race[race]
        draw_venn(axes[i], ae_set, man_set, race, ae_color, shared_color)

    axes[0].annotate(
        f"({panel_letter})\n{panel_title}",
        xy=(-0.38, 0.5), xycoords="axes fraction",
        fontsize=9.5, fontweight="bold", va="center", ha="center",
        rotation=90, annotation_clip=False, color=COLOR_TEXT,
    )


def main():
    xlsx = ROOT / "final_results_seed42.xlsx"
    data = load_data(xlsx)

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Autoencoder-Derived Core vs Manual Reference Core",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.955,
        "Per-race and overall overlap between AE-derived and manual core taxa",
        ha="center", fontsize=10.5, color="#444444",
    )

    gs = gridspec.GridSpec(
        2, 5, figure=fig,
        hspace=0.60, wspace=0.22,
        left=0.07, right=0.99, top=0.92, bottom=0.05,
    )

    draw_row(fig, gs, 0,
             data["AE_Core_119"]["ae"], data["AE_Core_119"]["man"],
             COLOR_AE_119, "A", "AE-119  (95% mass)",
             shared_color=COLOR_SHARED_119)

    draw_row(fig, gs, 1,
             data["AE_Core_70"]["ae"],  data["AE_Core_70"]["man"],
             COLOR_AE_70,  "B", "AE-70  (Score > mean)")

    out_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig_venn_ae_cores.pdf"
    png_path = out_dir / "fig_venn_ae_cores.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    sub_pdf = ROOT / "submission" / "figures" / "pdf"
    sub_png = ROOT / "submission" / "figures" / "png"
    shutil.copy(pdf_path, sub_pdf / "fig_venn_ae_cores.pdf")
    shutil.copy(png_path, sub_png / "fig_venn_ae_cores.png")
    print("Copied to submission/figures/pdf/ and submission/figures/png/")


if __name__ == "__main__":
    main()
