"""
Hierarchical PyTorch classifier for variable star taxonomy.

Implements a shared-backbone MLP with per-node classification heads that
follow the timedomain-taxonomy (tdtax) hierarchy.  Training uses masked
per-head losses so each head only receives gradients from sources that
belong to its branch.  Inference cascades top-down through the tree.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset, random_split

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Taxonomy node
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TaxonomyNode:
    """One node in the classification hierarchy."""

    name: str
    classes: list[str]
    label_map: dict[str, list[str] | None]  # class_name → label columns
    parent: str | None = None
    parent_class: str | None = None
    children: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.classes)


# ──────────────────────────────────────────────────────────────────────
# PyTorch model
# ──────────────────────────────────────────────────────────────────────


class HierarchicalNet(nn.Module):
    """Shared-backbone MLP with per-node classification heads."""

    def __init__(
        self,
        n_features: int,
        backbone_dims: list[int],
        dropout: list[float],
        nodes: dict[str, TaxonomyNode],
        quality_flags: list[str],
        batch_norm: bool = True,
    ):
        super().__init__()

        # Build backbone
        layers = []
        in_dim = n_features
        for i, out_dim in enumerate(backbone_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout[i] if i < len(dropout) else dropout[-1]))
            in_dim = out_dim
        self.backbone = nn.Sequential(*layers)

        # Classification heads (multi-class via softmax)
        self.heads = nn.ModuleDict()
        for name, node in nodes.items():
            self.heads[name] = nn.Linear(in_dim, node.n_classes)

        # Quality flag heads (binary via sigmoid)
        self.flag_heads = nn.ModuleDict()
        for flag in quality_flags:
            self.flag_heads[flag] = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw logits for every head."""
        z = self.backbone(x)
        out = {}
        for name, head in self.heads.items():
            out[name] = head(z)
        for name, head in self.flag_heads.items():
            out[name] = head(z)
        return out


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────


class HierarchyDataset(Dataset):
    """Holds feature tensor + per-head labels + masks."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ):
        self.features = features
        self.labels = labels
        self.masks = masks
        self.n = features.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        item = {"features": self.features[idx]}
        for k in self.labels:
            item[f"label_{k}"] = self.labels[k][idx]
            item[f"mask_{k}"] = self.masks[k][idx]
        return item


# ──────────────────────────────────────────────────────────────────────
# Main classifier wrapper
# ──────────────────────────────────────────────────────────────────────


class HierarchicalClassifier:
    """End-to-end training / inference wrapper."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Build feature column list
        self.feature_columns: list[str] = []
        for group in self.cfg["features"].values():
            self.feature_columns.extend(group["columns"])

        # Build taxonomy tree
        self.nodes: dict[str, TaxonomyNode] = {}
        tax = self.cfg["taxonomy"]
        qf_key = "quality_flags"

        for name, spec in tax.items():
            if name == qf_key:
                continue
            node = TaxonomyNode(
                name=name,
                classes=spec["classes"],
                label_map=spec.get("label_map", {}),
                parent=spec.get("parent"),
                parent_class=spec.get("parent_class"),
            )
            self.nodes[name] = node

        # Wire children
        for name, node in self.nodes.items():
            if node.parent and node.parent in self.nodes:
                self.nodes[node.parent].children.append(name)

        # Quality flags
        self.quality_flags: list[str] = []
        if qf_key in tax:
            self.quality_flags = list(tax[qf_key].keys())
            self._qf_columns = {
                k: v["label_columns"] for k, v in tax[qf_key].items()
            }

        # Model / normalization state (set during train or load)
        self.model: Optional[HierarchicalNet] = None
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.class_weights: dict[str, torch.Tensor] = {}
        self.device = torch.device("cpu")

    # ── feature extraction ────────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Pull feature columns, fill NaN with 0, return float32 array."""
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            log.warning("Missing %d feature columns, filling with NaN: %s",
                        len(missing), missing[:10])
        X = np.zeros((len(df), len(self.feature_columns)), dtype=np.float32)
        for i, col in enumerate(self.feature_columns):
            if col in df.columns:
                X[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        return X

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standardize features.  If fit=True, compute and store stats."""
        if fit:
            self.feat_mean = np.nanmean(X, axis=0).astype(np.float32)
            self.feat_std = np.nanstd(X, axis=0).astype(np.float32)
            self.feat_std[self.feat_std < 1e-8] = 1.0  # avoid div-by-zero
        X = (X - self.feat_mean) / self.feat_std
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # ── label preparation ─────────────────────────────────────────────

    def prepare_labels(
        self, df: pd.DataFrame, threshold: float = 0.5
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Convert multi-label confidence columns → integer class per node.

        Returns (labels, masks) where:
          labels[node_name] = int array of shape (N,)
          masks[node_name]  = bool array of shape (N,), True if this source
                              should contribute to that node's loss.
        """
        N = len(df)
        labels: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}

        # ── Step 1: compute per-class scores for each node ────────────
        node_scores: dict[str, np.ndarray] = {}  # (N, n_classes)
        for name, node in self.nodes.items():
            scores = np.zeros((N, node.n_classes), dtype=np.float32)
            for ci, cls in enumerate(node.classes):
                cols = node.label_map.get(cls)
                if cols is None:
                    continue
                for col in cols:
                    if col in df.columns:
                        vals = pd.to_numeric(
                            df[col], errors="coerce"
                        ).fillna(0).values
                        scores[:, ci] = np.maximum(scores[:, ci], vals)
            node_scores[name] = scores

        # ── Step 2: bottom-up propagation ─────────────────────────────
        # If a leaf label is set, force all ancestors to be consistent.
        # Process nodes from leaves to root.
        ordered = self._topo_sort_leaves_first()
        for name in ordered:
            node = self.nodes[name]
            if not node.parent:
                continue
            parent = self.nodes[node.parent]
            parent_ci = parent.classes.index(node.parent_class)
            # Any source with score >= threshold in *any* child class
            # should have the parent gating class forced on.
            child_max = node_scores[name].max(axis=1)
            active = child_max >= threshold
            cur = node_scores[parent.name][:, parent_ci]
            node_scores[parent.name][:, parent_ci] = np.where(
                active & (cur < threshold), threshold, cur
            )

        # ── Step 3: winner-take-all among siblings ────────────────────
        for name, node in self.nodes.items():
            scores = node_scores[name]
            winner = scores.argmax(axis=1)
            has_any = scores.max(axis=1) >= threshold
            labels[name] = winner
            # For the "remainder" class pattern (e.g., other_sawtooth),
            # if no class reaches threshold but the source is in this
            # branch, we assign the last class as the fallback.
            # The mask will handle whether this source is active.

            # ── Step 4: compute masks ─────────────────────────────────
            if node.parent is None:
                # Root node: every source participates
                masks[name] = np.ones(N, dtype=bool)
            else:
                parent = self.nodes[node.parent]
                parent_ci = parent.classes.index(node.parent_class)
                # Source must be (a) in the parent branch, and (b) parent
                # label matches the gating class.
                masks[name] = masks[parent.name] & (labels[parent.name] == parent_ci)

            # Sources in the branch but with no class above threshold:
            # assign to argmax anyway (soft label) but only if in branch
            # Mark non-branch sources as label=0 (doesn't matter, masked out)
            labels[name] = np.where(masks[name], winner, 0)

        # ── Step 5: quality flags ─────────────────────────────────────
        for flag in self.quality_flags:
            cols = self._qf_columns[flag]
            score = np.zeros(N, dtype=np.float32)
            for col in cols:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
                    score = np.maximum(score, vals)
            labels[flag] = (score >= threshold).astype(np.int64)
            masks[flag] = np.ones(N, dtype=bool)

        return labels, masks

    def _topo_sort_leaves_first(self) -> list[str]:
        """Return node names from leaves to root."""
        visited = set()
        order = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for child in self.nodes[name].children:
                visit(child)
            order.append(name)

        for name in self.nodes:
            visit(name)
        return order

    # ── dataset construction ──────────────────────────────────────────

    def build_dataset(
        self, df: pd.DataFrame, fit_norm: bool = False, threshold: float = 0.5
    ) -> HierarchyDataset:
        X = self._extract_features(df)
        X = self._normalize(X, fit=fit_norm)
        labels, masks = self.prepare_labels(df, threshold=threshold)

        feat_t = torch.from_numpy(X)
        labels_t = {k: torch.from_numpy(v.astype(np.int64)) for k, v in labels.items()}
        masks_t = {k: torch.from_numpy(v) for k, v in masks.items()}

        return HierarchyDataset(feat_t, labels_t, masks_t)

    # ── training ──────────────────────────────────────────────────────

    def _compute_class_weights(self, dataset: HierarchyDataset):
        """Inverse-frequency class weights for each head."""
        self.class_weights = {}
        for name, node in self.nodes.items():
            lab = dataset.labels[name].numpy()
            msk = dataset.masks[name].numpy()
            active_labels = lab[msk]
            if len(active_labels) == 0:
                self.class_weights[name] = torch.ones(node.n_classes)
                continue
            counts = np.bincount(active_labels, minlength=node.n_classes).astype(float)
            counts[counts == 0] = 1.0
            w = len(active_labels) / (node.n_classes * counts)
            self.class_weights[name] = torch.from_numpy(w.astype(np.float32))

    def train(
        self,
        df: pd.DataFrame,
        epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        val_fraction: float | None = None,
        output_dir: str | Path | None = None,
    ) -> dict:
        """Train the model.  Returns dict of final metrics."""
        tcfg = self.cfg["training"]
        epochs = epochs or tcfg["epochs"]
        lr = lr or tcfg["learning_rate"]
        batch_size = batch_size or tcfg["batch_size"]
        val_fraction = val_fraction or tcfg["val_fraction"]
        threshold = tcfg["label_threshold"]
        patience = tcfg.get("early_stopping_patience", 15)
        warmup_epochs = tcfg.get("warmup_epochs", 5)
        head_weights = tcfg.get("head_weights", {})

        log.info("Building dataset (%d sources) ...", len(df))
        full_ds = self.build_dataset(df, fit_norm=True, threshold=threshold)
        self._compute_class_weights(full_ds)

        # Train / val split
        n_val = int(len(full_ds) * val_fraction)
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

        # Build model
        mcfg = self.cfg["model"]
        n_features = len(self.feature_columns)
        self.model = HierarchicalNet(
            n_features=n_features,
            backbone_dims=mcfg["backbone_dims"],
            dropout=mcfg["dropout"],
            nodes=self.nodes,
            quality_flags=self.quality_flags,
            batch_norm=mcfg.get("batch_norm", True),
        ).to(self.device)

        # Loss functions (per-head)
        ce_losses = {}
        for name in self.nodes:
            w = self.class_weights[name].to(self.device)
            ce_losses[name] = nn.CrossEntropyLoss(weight=w, reduction="none")
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.cfg["training"]["weight_decay"],
        )

        # Cosine schedule with linear warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        log.info(
            "Training: %d epochs, lr=%.1e, batch=%d, train=%d, val=%d",
            epochs, lr, batch_size, n_train, n_val,
        )

        for epoch in range(epochs):
            # ── training step ─────────────────────────────────────────
            self.model.train()
            train_loss_sum = 0.0
            train_n = 0
            for batch in train_loader:
                x = batch["features"].to(self.device)
                logits = self.model(x)
                loss = self._compute_loss(
                    logits, batch, ce_losses, bce_loss, head_weights
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                train_loss_sum += loss.item() * x.shape[0]
                train_n += x.shape[0]
            scheduler.step()

            # ── validation step ───────────────────────────────────────
            self.model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["features"].to(self.device)
                    logits = self.model(x)
                    loss = self._compute_loss(
                        logits, batch, ce_losses, bce_loss, head_weights
                    )
                    val_loss_sum += loss.item() * x.shape[0]
                    val_n += x.shape[0]

            train_loss = train_loss_sum / max(train_n, 1)
            val_loss = val_loss_sum / max(val_n, 1)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                    epoch + 1, epochs, train_loss, val_loss,
                    scheduler.get_last_lr()[0],
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # Final evaluation on validation set
        metrics = self.evaluate_loader(val_loader)
        log.info("Final validation metrics:")
        for node_name, m in metrics.items():
            log.info("  %s: acc=%.3f  bal_acc=%.3f", node_name, m["accuracy"], m["balanced_accuracy"])

        # Save if requested
        if output_dir:
            self.save(output_dir)

        return metrics

    def _compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        ce_losses: dict[str, nn.CrossEntropyLoss],
        bce_loss: nn.BCEWithLogitsLoss,
        head_weights: dict[str, float],
    ) -> torch.Tensor:
        """Combined masked loss across all heads."""
        total = torch.tensor(0.0, device=self.device)

        for name in self.nodes:
            lab = batch[f"label_{name}"].to(self.device)
            msk = batch[f"mask_{name}"].to(self.device)
            n_active = msk.sum().item()
            if n_active == 0:
                continue
            per_sample = ce_losses[name](logits[name], lab)
            head_loss = (per_sample * msk.float()).sum() / n_active
            w = head_weights.get(name, 1.0)
            total = total + w * head_loss

        for flag in self.quality_flags:
            lab = batch[f"label_{flag}"].to(self.device).float()
            msk = batch[f"mask_{flag}"].to(self.device)
            n_active = msk.sum().item()
            if n_active == 0:
                continue
            per_sample = bce_loss(logits[flag].squeeze(-1), lab)
            head_loss = (per_sample * msk.float()).sum() / n_active
            w = head_weights.get(flag, 0.5)
            total = total + w * head_loss

        return total

    # ── inference ─────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cascading top-down inference.

        Returns DataFrame with columns:
          - pred_{node}: predicted class name at each level
          - prob_{node}: probability of predicted class
          - full_path: dot-separated path from root to leaf
          - flag_{name}: boolean quality flags
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        X = self._extract_features(df)
        X = self._normalize(X)
        x_t = torch.from_numpy(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_t)

        N = len(df)
        results: dict[str, list] = {}

        # Get softmax probs for each head
        probs: dict[str, np.ndarray] = {}
        for name, node in self.nodes.items():
            p = torch.softmax(logits[name], dim=1).cpu().numpy()
            probs[name] = p

        # Cascade: root → leaves
        preds: dict[str, np.ndarray] = {}  # integer predictions
        ordered = self._topo_sort_root_first()
        for name in ordered:
            node = self.nodes[name]
            p = probs[name]
            pred = p.argmax(axis=1)

            if node.parent is not None:
                # Mask: only predict for sources where parent predicted
                # the gating class
                parent = self.nodes[node.parent]
                parent_ci = parent.classes.index(node.parent_class)
                parent_match = preds[node.parent] == parent_ci
                # For sources not in this branch, set pred to -1
                pred = np.where(parent_match, pred, -1)

            preds[name] = pred

        # Build output columns
        for name, node in self.nodes.items():
            pred_names = []
            pred_probs = []
            for i in range(N):
                pi = preds[name][i]
                if pi >= 0:
                    pred_names.append(node.classes[pi])
                    pred_probs.append(float(probs[name][i, pi]))
                else:
                    pred_names.append("")
                    pred_probs.append(0.0)
            results[f"pred_{name}"] = pred_names
            results[f"prob_{name}"] = pred_probs

        # Full path
        paths = []
        for i in range(N):
            parts = []
            for name in ordered:
                pi = preds[name][i]
                if pi >= 0:
                    parts.append(self.nodes[name].classes[pi])
            paths.append(".".join(parts))
        results["full_path"] = paths

        # Quality flags
        for flag in self.quality_flags:
            flag_prob = torch.sigmoid(logits[flag]).squeeze(-1).cpu().numpy()
            results[f"flag_{flag}"] = flag_prob >= 0.5
            results[f"prob_{flag}"] = flag_prob.tolist()

        out = pd.DataFrame(results, index=df.index)
        return out

    def _topo_sort_root_first(self) -> list[str]:
        """Return node names from root to leaves (BFS order)."""
        roots = [n for n, node in self.nodes.items() if node.parent is None]
        order = []
        queue = list(roots)
        while queue:
            name = queue.pop(0)
            order.append(name)
            for child in self.nodes[name].children:
                queue.append(child)
        return order

    # ── evaluation ────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict[str, dict]:
        """Evaluate on a DataFrame, return per-node metrics."""
        threshold = self.cfg["training"]["label_threshold"]
        ds = self.build_dataset(df, fit_norm=False, threshold=threshold)
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        return self.evaluate_loader(loader)

    def evaluate_loader(self, loader: DataLoader) -> dict[str, dict]:
        """Evaluate on a DataLoader, return per-node metrics."""
        self.model.eval()
        all_labels: dict[str, list] = {n: [] for n in list(self.nodes) + self.quality_flags}
        all_preds: dict[str, list] = {n: [] for n in list(self.nodes) + self.quality_flags}
        all_masks: dict[str, list] = {n: [] for n in list(self.nodes) + self.quality_flags}

        with torch.no_grad():
            for batch in loader:
                x = batch["features"].to(self.device)
                logits = self.model(x)

                for name in self.nodes:
                    pred = logits[name].argmax(dim=1).cpu().numpy()
                    lab = batch[f"label_{name}"].numpy()
                    msk = batch[f"mask_{name}"].numpy()
                    all_preds[name].append(pred)
                    all_labels[name].append(lab)
                    all_masks[name].append(msk)

                for flag in self.quality_flags:
                    pred = (torch.sigmoid(logits[flag]).squeeze(-1) >= 0.5).int().cpu().numpy()
                    lab = batch[f"label_{flag}"].numpy()
                    msk = batch[f"mask_{flag}"].numpy()
                    all_preds[flag].append(pred)
                    all_labels[flag].append(lab)
                    all_masks[flag].append(msk)

        metrics = {}
        for name in list(self.nodes) + self.quality_flags:
            pred = np.concatenate(all_preds[name])
            lab = np.concatenate(all_labels[name])
            msk = np.concatenate(all_masks[name])

            active = msk.astype(bool)
            if active.sum() == 0:
                metrics[name] = {"accuracy": 0.0, "balanced_accuracy": 0.0, "n_active": 0}
                continue

            y_true = lab[active]
            y_pred = pred[active]

            acc = (y_true == y_pred).mean()
            bal_acc = balanced_accuracy_score(y_true, y_pred)

            if name in self.nodes:
                target_names = self.nodes[name].classes
            else:
                target_names = ["negative", "positive"]

            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(
                y_true, y_pred, target_names=target_names,
                output_dict=True, zero_division=0,
            )

            metrics[name] = {
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "n_active": int(active.sum()),
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
            }

        return metrics

    # ── save / load ───────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save model state, config, and normalization stats."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "model.pt")
        np.savez(
            path / "norm_stats.npz",
            mean=self.feat_mean,
            std=self.feat_std,
        )
        # Save class weights
        cw = {k: v.numpy() for k, v in self.class_weights.items()}
        np.savez(path / "class_weights.npz", **cw)
        # Copy config
        import shutil
        shutil.copy2(self.config_path, path / "hierarchy_config.yaml")
        log.info("Model saved to %s", path)

    def load(self, path: str | Path):
        """Load model state, config, and normalization stats."""
        path = Path(path)

        # Re-read config from saved copy
        saved_cfg = path / "hierarchy_config.yaml"
        if saved_cfg.exists():
            with open(saved_cfg) as f:
                self.cfg = yaml.safe_load(f)

        # Rebuild feature list + taxonomy (in case config changed)
        self.__init__(saved_cfg if saved_cfg.exists() else self.config_path)

        # Normalization stats
        stats = np.load(path / "norm_stats.npz")
        self.feat_mean = stats["mean"]
        self.feat_std = stats["std"]

        # Class weights
        cw_path = path / "class_weights.npz"
        if cw_path.exists():
            cw = np.load(cw_path)
            self.class_weights = {k: torch.from_numpy(cw[k]) for k in cw.files}

        # Build and load model
        mcfg = self.cfg["model"]
        self.model = HierarchicalNet(
            n_features=len(self.feature_columns),
            backbone_dims=mcfg["backbone_dims"],
            dropout=mcfg["dropout"],
            nodes=self.nodes,
            quality_flags=self.quality_flags,
            batch_norm=mcfg.get("batch_norm", True),
        )
        self.model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        log.info("Model loaded from %s", path)
