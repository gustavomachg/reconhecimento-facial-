"""Fine-tuning da FaceNet guiado por config (E03, E04, E05).

    python -m facerec.train --config configs/experiments/E03_facenet_partial_triplet.yaml

Estratégias (`training.strategy`):
- partial: congela a rede e treina só os blocos de `training.trainable_blocks`
           (os últimos blocos, mais específicos da tarefa). Menos parâmetros -> menos
           risco de overfitting com poucos dados e treino mais barato.
- full:    treina todos os blocos (normalmente com LR menor).

Losses (`loss.name`): triplet (batch-hard, batches P×K) ou arcface (cabeça de margem angular).

Validação a cada época, em identidades DISJUNTAS do treino:
- val_loss: Triplet batch-hard nos embeddings da validação (mesma medida para as duas
            losses, para que as curvas sejam comparáveis entre E03/E04/E05);
- val_eer / val_auc: verificação em pares da validação.
Early stopping pelo val_eer; o melhor checkpoint vai para `checkpoints/` (ignorado pelo git).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from facerec.config import environment_info, load_yaml, pick_device, resolve_path, set_seed
from facerec.data.datasets import FaceImageDataset, list_identity_folder, make_verification_pairs
from facerec.data.samplers import PKBatchSampler
from facerec.data.splits import filter_records, split_by_identity
from facerec.data.transforms import build_transforms
from facerec.losses import TripletLoss, build_loss
from facerec.metrics import compute_eer, roc_auc
from facerec.models.facenet import FACENET_BLOCKS, build_facenet


def apply_exclusions(records: list, data_cfg: dict) -> list:
    """Remove identidades suspeitas de sobreposição com o LFW (check_identity_overlap.py)
    e quase-duplicatas entre conjuntos (check_duplicates.py), se as listas existirem."""
    ids_file = data_cfg.get("exclude_identities_file")
    if ids_file and resolve_path(ids_file).exists():
        bad = {
            x if x.startswith("celeba_") else f"celeba_{x}"
            for x in resolve_path(ids_file).read_text(encoding="utf-8").split()
        }
        records = [r for r in records if r.identity not in bad]
    dup_file = data_cfg.get("exclude_files_list")
    if dup_file and resolve_path(dup_file).exists():
        bad_paths = set(resolve_path(dup_file).read_text(encoding="utf-8").splitlines())
        records = [r for r in records if r.path not in bad_paths]
    return records


def configure_trainable(model: torch.nn.Module, strategy: str, blocks: list[str]) -> list[str]:
    """Define quais blocos recebem gradiente. Retorna a lista de blocos treináveis."""
    if strategy == "full":
        # `logits` é o classificador do VGGFace2: não participa do embedding, fica de fora
        blocks = [n for n, _ in model.named_children() if n != "logits"]
    elif strategy != "partial":
        raise ValueError("training.strategy deve ser 'partial' ou 'full'")
    unknown = set(blocks) - set(FACENET_BLOCKS) - {n for n, _ in model.named_children()}
    if unknown:
        raise ValueError(f"Blocos inexistentes: {unknown}")
    for name, module in model.named_children():
        for p in module.parameters():
            p.requires_grad = name in blocks
    return blocks


def set_train_mode(model: torch.nn.Module, trainable: list[str]) -> None:
    """model.train(), mas blocos CONGELADOS ficam em eval: suas BatchNorms mantêm as
    estatísticas do pré-treino em vez de se adaptarem a batches pequenos."""
    model.train()
    for name, module in model.named_children():
        if name not in trainable:
            module.eval()


def count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    """(treináveis, total) — sem a camada `logits` do VGGFace2, que não gera o embedding."""
    params = [p for n, p in model.named_parameters() if not n.startswith("logits")]
    return int(sum(p.numel() for p in params if p.requires_grad)), int(
        sum(p.numel() for p in params)
    )


@torch.no_grad()
def embed_dataset(model, dataset, device, batch_size=128, num_workers=0) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = [model(x.to(device)).cpu() for x, _ in loader]
    return torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()


def validate(model, val_ds, val_pairs, device, num_workers=0, seed=0) -> dict:
    emb = embed_dataset(model, val_ds, device, num_workers=num_workers)
    labels = torch.tensor(val_ds.labels)
    # val_loss: média da triplet batch-hard em batches P×K FIXOS (seed fixa)
    sampler = PKBatchSampler(val_ds.labels, p=16, k=4, seed=seed)
    trip = TripletLoss(margin=0.2)
    e = torch.from_numpy(emb)
    val_loss = float(np.mean([trip(e[b], labels[b]).item() for b in sampler]))
    index = {r.path: i for i, r in enumerate(val_ds.records)}
    s = np.array([float(emb[index[a]] @ emb[index[b]]) for a, b, _ in val_pairs])
    y = np.array([lbl for _, _, lbl in val_pairs])
    eer, _ = compute_eer(s, y)
    return {"val_loss": val_loss, "val_eer": eer, "val_auc": roc_auc(s, y)}


def save_curves(history: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="treino (loss do exp.)")
    ax[0].plot(ep, [h["val_loss"] for h in history], marker="o", label="validação (triplet)")
    ax[0].set(xlabel="época", ylabel="loss", title="Curvas de loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_eer"] for h in history], marker="o", color="tab:red")
    ax[1].set(xlabel="época", ylabel="EER", title="EER na validação")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)


def train(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    exp, data_cfg, tr, loss_cfg = cfg["experiment"], cfg["data"], cfg["training"], cfg["loss"]
    set_seed(cfg.get("seed", 42))
    device = pick_device(cfg.get("device", "auto"))
    out_dir = resolve_path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config_used.yaml")

    # ---------------- dados: split por identidade ANTES de qualquer augmentation
    records = list_identity_folder(resolve_path(data_cfg["aligned_root"]))
    records = apply_exclusions(records, data_cfg)
    split = split_by_identity(
        [r.identity for r in records],
        data_cfg.get("val_fraction", 0.15),
        data_cfg.get("test_fraction", 0.15),
        seed=cfg.get("seed", 42),
    )
    train_recs = filter_records(records, split["train"])
    val_recs = filter_records(records, split["val"])
    size = data_cfg.get("image_size", 160)
    train_ds = FaceImageDataset(train_recs, build_transforms(True, size, data_cfg.get("augment")))
    val_ds = FaceImageDataset(val_recs, build_transforms(False, size))
    val_pairs = make_verification_pairs(
        val_recs, data_cfg.get("val_pairs", 3000), seed=cfg.get("seed", 42), cross_session=False
    )
    (out_dir / "split.json").write_text(
        json.dumps(split)
    )  # IDs do CelebA (dataset público, encoding="utf-8")

    # ---------------- modelo, loss, otimizador
    model = build_facenet(pretrained=cfg["model"].get("pretrained", "vggface2")).to(device)
    trainable = configure_trainable(
        model, tr["strategy"], tr.get("trainable_blocks", ["block8", "last_linear", "last_bn"])
    )
    n_train_params, n_params = count_trainable(model)
    criterion = build_loss(loss_cfg, n_classes=len(train_ds.label_map)).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(criterion.parameters())
    opt_name = tr.get("optimizer", "adam").lower()
    lr, wd = float(tr["lr"]), float(tr.get("weight_decay", 0.0))
    if opt_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer desconhecido: {opt_name}")
    epochs = int(tr["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # precisão mista (AMP) na GPU: ~metade da memória e mais rápido; a loss fica em float32
    use_amp = device.startswith("cuda") and tr.get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    nw = data_cfg.get("num_workers", 0)
    if loss_cfg["name"] == "triplet":
        pk = tr.get("batch", {"p": 16, "k": 4})
        sampler = PKBatchSampler(train_ds.labels, pk["p"], pk["k"], seed=cfg.get("seed", 42))
        loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=nw)
    else:
        loader = DataLoader(
            train_ds,
            batch_size=tr.get("batch_size", 64),
            shuffle=True,
            drop_last=True,
            num_workers=nw,
        )

    ckpt_path = resolve_path(cfg["output"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history, bad_epochs = [], 0
    patience = tr.get("early_stopping_patience", 5)

    base = validate(model, val_ds, val_pairs, device, nw)
    history.append({"epoch": 0, "train_loss": float("nan"), **base, "lr": lr})
    print(f"[época 0] pré-treinado: {base}")
    # o ponto de partida também é candidato: se nenhuma época superar o pré-treinado,
    # o checkpoint final é o próprio pré-treinado (e isso é registrado em best_epoch=0)
    best_eer, best_epoch = base["val_eer"], 0
    torch.save({"model": model.state_dict(), "epoch": 0}, ckpt_path)
    last_path = ckpt_path.with_name(ckpt_path.stem + "_last" + ckpt_path.suffix)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        set_train_mode(model, trainable)
        criterion.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                emb = model(x)
            loss = criterion(emb.float(), y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        val = validate(model, val_ds, val_pairs, device, nw)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **val,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(f"[época {epoch}] {row}")
        # checkpoint da ÚLTIMA época: permite medir no benchmark o efeito do treino mesmo
        # quando o early stopping fica com o pré-treinado (análise de overfitting)
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)
        if val["val_eer"] < best_eer:
            best_eer, best_epoch, bad_epochs = val["val_eer"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping na época {epoch}")
                break
    if best_epoch == 0:
        print("Aviso: nenhuma época superou o modelo pré-treinado na validação.")

    summary = {
        "experiment": exp,
        "strategy": tr["strategy"],
        "trainable_blocks": trainable,
        "loss": loss_cfg,
        "n_parameters": n_params,
        "n_trainable_parameters": n_train_params,
        "n_train_identities": len(split["train"]),
        "n_val_identities": len(split["val"]),
        "n_train_images": len(train_recs),
        "n_val_images": len(val_recs),
        "best_val_eer": best_eer,
        "best_epoch": best_epoch,
        "pretrained_val_eer": base["val_eer"],
        "train_time_s": round(time.time() - t0, 1),
        "device": device,
        "amp": use_amp,
        "history": history,
        "environment": environment_info(),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    save_curves(history, out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tuning da FaceNet")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    s = train(args.config)
    print(json.dumps({k: v for k, v in s.items() if k != "history"}, indent=2, default=str))


if __name__ == "__main__":
    main()
