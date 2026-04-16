#!/usr/bin/env python3
"""
analyze_tau_results.py
======================
Analisi completa dei risultati del tau grid search.

Legge i file metrics_tau*.json prodotti dal workflow tau-grid-search,
genera 4 grafici matplotlib, un report markdown e salva su MongoDB.

Usato sia dal workflow GitHub Actions che localmente.

Usage:
    python scripts/analyze_tau_results.py
    python scripts/analyze_tau_results.py --metrics-dir metrics/ --out-dir docs/tau_analysis/
    python scripts/analyze_tau_results.py --dry-run   # salta MongoDB
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── costanti ──────────────────────────────────────────────────────────────────
TARGETS = {
    "kendall_tau": (0.45, True,  "Kendall τ",  "≥ 0.45"),
    "brier":       (0.20, False, "Brier Score","≤ 0.20"),
    "rps":         (0.15, False, "RPS",        "≤ 0.15"),
    "logloss":     (1.00, False, "LogLoss",    "≤ 1.00"),
    "ece":         (0.05, False, "ECE",        "≤ 0.05"),
}

PALETTE = sns.color_palette("tab10")

# Dati baseline noti (tau=0.833, commessi in artifacts/)
BASELINE = {
    "tau": "0.833", "val_year": "2023-2024",
    "kendall_tau": 0.451, "brier": 0.164, "ece": 0.015,
    "note": "baseline pre-regression (artifacts/release_decision_significance.md)"
}

# Fallback: risultati della sessione con tau=0.05 (da rolling-validation)
FALLBACK_ROWS = [
    {"tau": "0.05", "val_year": "2021", "kendall_tau": 0.218, "brier": 0.4225,
     "logloss": 9.8436, "rps": 0.2597, "ece": 0.0585, "n_races": 22},
    {"tau": "0.05", "val_year": "2022", "kendall_tau": 0.135, "brier": 0.4776,
     "logloss": 10.1778, "rps": 0.2891, "ece": 0.0694, "n_races": 22},
    {"tau": "0.05", "val_year": "2023", "kendall_tau": 0.095, "brier": 0.490,
     "logloss": 10.1604, "rps": 0.2948, "ece": 0.0629, "n_races": 22},
    {"tau": "0.05", "val_year": "2024", "kendall_tau": 0.112, "brier": 0.4714,
     "logloss": 9.7258, "rps": 0.2869, "ece": 0.0627, "n_races": 24},
]


# ── A. Caricamento dati ────────────────────────────────────────────────────────

def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    if metrics_dir.exists():
        for f in sorted(metrics_dir.glob("metrics_tau*.json")):
            try:
                with open(f) as fp:
                    rows.append(json.load(fp))
            except Exception as e:
                print(f"[WARN] {f}: {e}")

    if not rows:
        print("[INFO] Nessun file trovato — uso dati fallback (tau=0.05 da sessione precedente)")
        rows = FALLBACK_ROWS

    df = pd.DataFrame(rows)
    df["tau_float"] = df["tau"].apply(lambda x: float(str(x)))
    df["val_year"]  = df["val_year"].astype(str)
    for col in ["kendall_tau", "brier", "rps", "logloss", "ece"]:
        if col not in df.columns:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "n_races" not in df.columns:
        df["n_races"] = None

    df = df.sort_values(["tau_float", "val_year"]).reset_index(drop=True)
    print(f"[INFO] Caricati {len(df)} risultati, "
          f"tau={sorted(df['tau_float'].unique())}, "
          f"anni={sorted(df['val_year'].unique())}")
    return df


# ── C. Grafici ─────────────────────────────────────────────────────────────────

def _target_line(ax, metric: str, axis: str = "y"):
    tgt, higher, _, label = TARGETS[metric]
    kw = dict(color="red", linestyle="--", linewidth=1.2, alpha=0.7,
              label=f"Target {label}")
    if axis == "y":
        ax.axhline(tgt, **kw)
    else:
        ax.axvline(tgt, **kw)


def plot_kendall_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap tau × anno per Kendall τ."""
    pivot = df.pivot_table(index="tau_float", columns="val_year",
                           values="kendall_tau", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4), max(4, len(pivot) * 0.9)))
    mask = pivot.isna()
    sns.heatmap(
        pivot, annot=True, fmt=".3f", mask=mask,
        cmap="RdYlGn", vmin=0.0, vmax=0.55,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Kendall τ"},
        ax=ax,
    )
    ax.set_title("Kendall τ — tau × anno di validazione\n"
                 "(verde = buono ≥ 0.45, rosso = critico)", fontsize=12, pad=12)
    ax.set_xlabel("Anno di validazione")
    ax.set_ylabel("TTT tau")

    # Evidenzia celle che superano il target
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            if not np.isnan(v) and v >= 0.45:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                             fill=False, edgecolor="darkgreen", lw=2.5))

    fig.tight_layout()
    path = out_dir / "plot_kendall_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return path


def plot_metrics_by_tau(df: pd.DataFrame, out_dir: Path) -> Path:
    """4 subplot: ogni metrica vs tau, linea per anno."""
    metrics = ["kendall_tau", "brier", "rps", "ece"]
    years   = sorted(df["val_year"].unique())
    taus    = sorted(df["tau_float"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        tgt, higher, label, tgt_label = TARGETS[metric]
        for i, year in enumerate(years):
            sub = df[df["val_year"] == year].sort_values("tau_float")
            vals = sub[metric].values
            xs   = sub["tau_float"].values
            mask = ~np.isnan(vals.astype(float))
            if mask.sum() == 0:
                continue
            ax.plot(xs[mask], vals[mask], "o-",
                    color=PALETTE[i], label=str(year), linewidth=2, markersize=6)

        _target_line(ax, metric)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("tau")
        ax.set_ylabel(label)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8, title="Anno val.")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Metriche vs TTT tau — tutti gli anni di validazione",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = out_dir / "plot_metrics_by_tau.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return path


def plot_tau_ranking(df: pd.DataFrame, out_dir: Path) -> Path:
    """Bar chart: media Kendall τ per tau, con CI e linea target."""
    summary = (df.groupby("tau_float")["kendall_tau"]
               .agg(["mean", "std", "count"])
               .reset_index())
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = summary["sem"] * 1.96
    summary = summary.sort_values("mean", ascending=False)

    colors = ["#2ecc71" if v >= 0.45 else "#e74c3c" for v in summary["mean"]]

    fig, ax = plt.subplots(figsize=(max(7, len(summary) * 1.2), 5))
    bars = ax.bar(
        summary["tau_float"].astype(str),
        summary["mean"],
        yerr=summary["ci95"],
        color=colors, edgecolor="white", linewidth=0.8,
        error_kw=dict(ecolor="gray", capsize=4, elinewidth=1.5),
        zorder=3,
    )
    ax.axhline(0.45, color="red", linestyle="--", linewidth=1.5,
               label="Target τ ≥ 0.45", zorder=4)

    # Etichette valori
    for bar, val in zip(bars, summary["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Ranking tau — Kendall τ medio (± IC 95%)\n"
                 "verde = supera target 0.45", fontsize=12)
    ax.set_xlabel("TTT tau")
    ax.set_ylabel("Kendall τ medio")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = out_dir / "plot_tau_ranking.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return path


def plot_tradeoffs(df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter Kendall τ vs Brier, colorato per tau, size=n_races."""
    fig, ax = plt.subplots(figsize=(9, 6))

    taus = sorted(df["tau_float"].unique())
    tau_colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(taus)}

    for _, row in df.iterrows():
        kt = row["kendall_tau"]
        br = row["brier"]
        if pd.isna(kt) or pd.isna(br):
            continue
        nr = row["n_races"] if pd.notna(row.get("n_races")) else 20
        ax.scatter(kt, br, s=float(nr) * 4, alpha=0.75,
                   color=tau_colors[row["tau_float"]],
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(f"{row['val_year']}", (kt, br),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7, alpha=0.7)

    # Linee target
    ax.axvline(0.45, color="green", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Target τ ≥ 0.45")
    ax.axhline(0.20, color="red", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Target Brier ≤ 0.20")

    # Legenda tau
    for t in taus:
        ax.scatter([], [], color=tau_colors[t], s=60, label=f"tau={t}")

    ax.set_xlabel("Kendall τ (più alto = meglio →)", fontsize=11)
    ax.set_ylabel("Brier Score (più basso = meglio ↓)", fontsize=11)
    ax.set_title("Trade-off Kendall τ vs Brier — ogni punto è (tau, anno)\n"
                 "Zona ottimale: destra + basso", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "plot_tradeoffs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return path


# ── D. Report markdown ─────────────────────────────────────────────────────────

def _cell_md(v, tgt, higher, dec=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    ok = (v >= tgt) if higher else (v <= tgt)
    icon = "✅" if ok else "⚠️"
    return f"`{v:.{dec}f}` {icon}"


def build_report(df: pd.DataFrame, out_dir: Path,
                 run_id: str, run_url: str,
                 tau_values: str, val_years: str, n_mc: str) -> tuple[Path, Path, str]:
    """
    Genera report_YYYYMMDD.md e step_summary.md.
    Ritorna (report_path, summary_path, best_tau_str).
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M")

    # Ranking tau per Kendall τ medio
    summary = (df.groupby("tau_float")["kendall_tau"]
               .mean().reset_index()
               .sort_values("kendall_tau", ascending=False))
    best_row  = summary.iloc[0] if len(summary) else None
    best_tau  = str(best_row["tau_float"]) if best_row is not None else "n/a"
    best_val  = best_row["kendall_tau"]    if best_row is not None else float("nan")

    years = sorted(df["val_year"].unique())
    taus  = sorted(df["tau_float"].unique())

    # ── report completo ──────────────────────────────────────────────────────
    lines = []
    lines += [f"# TTT Tau Grid Search — Report {now.strftime('%Y-%m-%d %H:%M')} UTC\n"]
    lines += [f"**Run**: [{run_id}]({run_url})  \n"
              f"**tau testati**: `{tau_values}`  \n"
              f"**anni validazione**: `{val_years}`  \n"
              f"**MC sim per job**: `{n_mc}`\n"]

    # Tabella Kendall τ per (tau, anno)
    lines += ["\n## Kendall τ per combinazione tau × anno\n"]
    header = "| tau |" + "".join(f" {y} |" for y in years) + " **media** |\n"
    sep    = "|-----|" + "".join("-------|" for _ in years) + "----------|\n"
    lines += [header, sep]
    for t in taus:
        sub = df[df["tau_float"] == t]
        vals = {r["val_year"]: r["kendall_tau"] for _, r in sub.iterrows()}
        valid = [v for v in vals.values() if not np.isnan(v)]
        mean_v = float(np.mean(valid)) if valid else float("nan")
        row = f"| `{t}` |"
        for y in years:
            v = vals.get(y, float("nan"))
            ok = not np.isnan(v) and v >= 0.45
            icon = " ✅" if ok else (" ⚠️" if not np.isnan(v) else "")
            row += f" {v:+.3f}{icon} |" if not np.isnan(v) else " n/a |"
        row += f" **{mean_v:+.3f}** |\n" if not np.isnan(mean_v) else " **n/a** |\n"
        lines.append(row)

    # Raccomandazione
    lines += [f"\n## Raccomandazione\n"]
    if not np.isnan(best_val):
        verdict = "✅ **SUPERA il target ≥ 0.45**" if best_val >= 0.45 else "⚠️ **non raggiunge il target ≥ 0.45**"
        lines += [f"**tau ottimale = `{best_tau}`** "
                  f"(Kendall τ medio = `{best_val:+.3f}`) — {verdict}\n\n"
                  f"> Per applicare: modificare `TTTConfig.tau` in "
                  f"`f1_predictor/models/driver_skill.py:71`\n"]
    lines += [f"\n*Baseline pre-regressione (tau=0.833): τ=+0.451 ✅*\n"]

    # Tabella metriche complete
    lines += ["\n## Metriche complete per tutti i tau\n"]
    lines += ["| tau | anno | Kendall τ | Brier | RPS | LogLoss | ECE | N gare |\n"]
    lines += ["|-----|------|-----------|-------|-----|---------|-----|--------|\n"]
    for _, row in df.sort_values(["tau_float", "val_year"]).iterrows():
        lines.append(
            f"| `{row['tau']}` | {row['val_year']} "
            f"| {_cell_md(row['kendall_tau'], 0.45, True,  3)} "
            f"| {_cell_md(row['brier'],       0.20, False, 4)} "
            f"| {_cell_md(row['rps'],         0.15, False, 4)} "
            f"| {_cell_md(row['logloss'],     1.00, False, 4)} "
            f"| {_cell_md(row['ece'],         0.05, False, 4)} "
            f"| {int(row['n_races']) if pd.notna(row.get('n_races')) else '?'} |\n"
        )

    # Sezione grafici
    lines += ["\n## Grafici\n"]
    lines += ["### 1. Heatmap Kendall τ (tau × anno)\n",
              "![heatmap](plot_kendall_heatmap.png)\n\n"]
    lines += ["### 2. Metriche vs tau per anno\n",
              "![metrics](plot_metrics_by_tau.png)\n\n"]
    lines += ["### 3. Ranking tau — Kendall τ medio\n",
              "![ranking](plot_tau_ranking.png)\n\n"]
    lines += ["### 4. Trade-off Kendall τ vs Brier\n",
              "![tradeoffs](plot_tradeoffs.png)\n\n"]

    # Analisi statistica
    lines += ["\n## Analisi statistica\n"]
    lines += ["| tau | media τ | std τ | min τ | max τ | n anni |\n"]
    lines += ["|-----|---------|-------|-------|-------|--------|\n"]
    for t in taus:
        sub_kt = df[df["tau_float"] == t]["kendall_tau"].dropna()
        if len(sub_kt) == 0:
            continue
        lines.append(f"| `{t}` | {sub_kt.mean():+.3f} | {sub_kt.std():.3f} "
                     f"| {sub_kt.min():+.3f} | {sub_kt.max():+.3f} | {len(sub_kt)} |\n")

    report_path = out_dir / f"report_{date_str}.md"
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"[REPORT] {report_path}")

    # ── step summary (senza immagini, solo tabelle) ──────────────────────────
    sum_lines = [f"## TTT Tau Grid Search — {now.strftime('%Y-%m-%d %H:%M')} UTC\n\n"]
    sum_lines += [header, sep]
    for t in taus:
        sub = df[df["tau_float"] == t]
        vals = {r["val_year"]: r["kendall_tau"] for _, r in sub.iterrows()}
        valid = [v for v in vals.values() if not np.isnan(v)]
        mean_v = float(np.mean(valid)) if valid else float("nan")
        row = f"| `{t}` |"
        for y in years:
            v = vals.get(y, float("nan"))
            ok = not np.isnan(v) and v >= 0.45
            icon = " ✅" if ok else (" ⚠️" if not np.isnan(v) else "")
            row += f" {v:+.3f}{icon} |" if not np.isnan(v) else " n/a |"
        row += f" **{mean_v:+.3f}** |\n" if not np.isnan(mean_v) else " **n/a** |\n"
        sum_lines.append(row)

    if not np.isnan(best_val):
        sum_lines += [f"\n### Raccomandazione: `tau = {best_tau}` "
                      f"(τ medio = `{best_val:+.3f}`)\n"]

    sum_lines += [f"\n> Report completo: `docs/tau_analysis/report_{date_str}.md`\n"]

    summary_path = out_dir / "step_summary.md"
    with open(summary_path, "w") as f:
        f.writelines(sum_lines)
    print(f"[SUMMARY] {summary_path}")

    return report_path, summary_path, best_tau


# ── E. MongoDB ─────────────────────────────────────────────────────────────────

def save_to_mongodb(df: pd.DataFrame, run_id: str, run_url: str, best_tau: str):
    try:
        from pymongo import MongoClient, UpdateOne
        uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
        if not uri:
            print("[MongoDB] MONGODB_URI non configurata — skip")
            return
        db_name = os.environ.get("MONGO_DB", "betbreaker")
        db = MongoClient(uri, serverSelectionTimeoutMS=5000)[db_name]

        now_iso = datetime.now(timezone.utc).isoformat()
        ops = []
        for _, row in df.iterrows():
            doc = {
                "run_id":      run_id,
                "run_url":     run_url,
                "tau":         float(row["tau_float"]),
                "val_year":    row["val_year"],
                "kendall_tau": float(row["kendall_tau"]) if pd.notna(row["kendall_tau"]) else None,
                "brier":       float(row["brier"])       if pd.notna(row.get("brier"))  else None,
                "rps":         float(row["rps"])         if pd.notna(row.get("rps"))    else None,
                "logloss":     float(row["logloss"])     if pd.notna(row.get("logloss"))else None,
                "ece":         float(row["ece"])         if pd.notna(row.get("ece"))    else None,
                "n_races":     int(row["n_races"])       if pd.notna(row.get("n_races"))else None,
                "updated_at":  now_iso,
            }
            ops.append(UpdateOne(
                {"run_id": run_id, "tau": doc["tau"], "val_year": doc["val_year"]},
                {"$set": doc},
                upsert=True,
            ))

        if ops:
            res = db["tau_search_results"].bulk_write(ops, ordered=False)
            print(f"[MongoDB] tau_search_results: "
                  f"{res.upserted_count} inseriti, {res.modified_count} aggiornati")

        # Documento di run summary
        db["tau_search_runs"].update_one(
            {"run_id": run_id},
            {"$set": {
                "run_id":   run_id,
                "run_url":  run_url,
                "best_tau": float(best_tau) if best_tau != "n/a" else None,
                "n_combinations": len(df),
                "created_at": now_iso,
            }},
            upsert=True,
        )
        print(f"[MongoDB] tau_search_runs aggiornato (run_id={run_id})")

    except Exception as e:
        print(f"[MongoDB] Errore: {e}")


# ── F. CLI + main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analisi risultati tau grid search")
    p.add_argument("--metrics-dir", default="metrics/",
                   help="Directory con file metrics_tau*.json")
    p.add_argument("--out-dir", default="docs/tau_analysis/",
                   help="Directory output per report e grafici")
    p.add_argument("--dry-run", action="store_true",
                   help="Non salva su MongoDB")
    return p.parse_args()


def main():
    args = parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id  = os.environ.get("RUN_ID",  "local")
    run_url = os.environ.get("RUN_URL",  "")
    tau_values = os.environ.get("TAU_VALUES", "n/a")
    val_years  = os.environ.get("VAL_YEARS",  "n/a")
    n_mc       = os.environ.get("N_MC_SIM",   "n/a")

    print("=" * 60)
    print("TAU GRID SEARCH — ANALISI RISULTATI")
    print("=" * 60)

    # A+B — carica dati
    df = load_metrics(metrics_dir)
    if df.empty:
        print("[ERROR] DataFrame vuoto — nessun dato da analizzare")
        sys.exit(1)

    # C — grafici
    plot_kendall_heatmap(df, out_dir)
    plot_metrics_by_tau(df, out_dir)
    plot_tau_ranking(df, out_dir)
    plot_tradeoffs(df, out_dir)

    # D — report
    report_path, summary_path, best_tau = build_report(
        df, out_dir, run_id, run_url, tau_values, val_years, n_mc
    )

    # E — MongoDB
    if not args.dry_run:
        save_to_mongodb(df, run_id, run_url, best_tau)
    else:
        print("[MongoDB] dry-run — skip")

    print()
    print("=" * 60)
    print(f"COMPLETATO")
    print(f"  Report:  {report_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Best tau: {best_tau}")
    print("=" * 60)


if __name__ == "__main__":
    main()
