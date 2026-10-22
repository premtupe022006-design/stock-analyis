"""Small helpers: output folders, figure/table saving, headings."""
import os
import matplotlib
matplotlib.use("Agg")  # headless: figures are saved to outputs/figures
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
TAB_DIR = os.path.join(ROOT, "outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=120)
    plt.close()


def save_table(df, name):
    df.to_csv(os.path.join(TAB_DIR, name + ".csv"))


def header(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
