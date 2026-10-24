"""Gera a tabela de resultados do README e o gráfico ROC comparativo a partir de
experiments/results/*/metrics.json (somente métricas agregadas).

    python -m facerec.report
"""

from __future__ import annotations

import json

from facerec.config import resolve_path

RESULTS = resolve_path("experiments/results")


def _fmt(v, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.2f}%" if pct else f"{v:.4f}"


def collect() -> dict[str, dict]:
    out = {}
    for f in sorted(RESULTS.glob("E0*/metrics.json")):
        out[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
    return out


def table(results: dict[str, dict]) -> str:
    lines = [
        "| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | EER | Observação |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for exp_id, r in results.items():
        e = r.get("experiment", {})
        model = (
            r.get("model") if isinstance(r.get("model"), str) else r.get("model", {}).get("name")
        )
        strategy = e.get("strategy", "")
        if "lfw" in r:
            lfw = r["lfw"]
            acc = f"{_fmt(lfw['accuracy_mean'])} ± {_fmt(lfw['accuracy_std'])}"
            lines.append(
                f"| {exp_id} | {model} | {strategy} | LFW (10-fold) | {acc} | "
                f"{_fmt(lfw['f1_mean'])} | {_fmt(lfw['eer'])} | AUC {_fmt(lfw['roc_auc'], False)} |"
            )
        if "private_val" in r:
            pv = r["private_val"]
            lines.append(
                f"| {exp_id} | {model} | {strategy} | Privado (val.) | {_fmt(pv['accuracy'])} | "
                f"{_fmt(pv['f1'])} | {_fmt(pv['eer'])} | AUC {_fmt(pv['roc_auc'], False)} |"
            )
        if "matchers" in r:
            for name, m in r["matchers"].items():
                lines.append(
                    f"| {exp_id} | {model} + {name} | open-set | Privado (val.) | "
                    f"{_fmt(m['accuracy'])} | {_fmt(m['macro_f1'])} | {_fmt(m.get('eer'))} | "
                    f"FAR desc. {_fmt(m['false_accept_unknown'])} |"
                )
        if "test" in r:
            t, tb = r["test"], r["test_binary_accept_reject"]
            lines.append(
                f"| {exp_id} | {model} + {r['matcher']['name']} | "
                f"thr={r['chosen_threshold']:.3f} | "
                f"Privado (TESTE) | {_fmt(t['accuracy'])} | {_fmt(t['macro_f1'])} | "
                f"{_fmt(tb.get('eer'))} | FAR {_fmt(tb['far'])} / FRR {_fmt(tb['frr'])} |"
            )
    return "\n".join(lines)


def roc_comparison(results: dict[str, dict], key: str = "lfw") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for exp_id, r in results.items():
        if key in r and "roc_curve" in r[key]:
            c = r[key]["roc_curve"]
            ax.plot(c["fpr"], c["tpr"], label=f"{exp_id} (AUC {r[key]['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(
        xscale="log",
        xlim=(1e-3, 1),
        xlabel="FAR (escala log)",
        ylabel="TAR = 1 - FRR",
        title=f"ROC comparativa — {key}",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"roc_comparison_{key}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    results = collect()
    if not results:
        print("Nenhum resultado em experiments/results/. Rode os experimentos primeiro.")
        return
    md = table(results)
    (RESULTS / "results_table.md").write_text(md + "\n", encoding="utf-8")
    for key in ("lfw", "private_val"):
        roc_comparison(results, key)
    print(md)


if __name__ == "__main__":
    main()
