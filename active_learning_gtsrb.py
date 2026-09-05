"""Pool-based active learning for traffic-sign recognition on GTSRB.

The labels of the unlabeled pool are hidden from the selection strategy and are
read only after a query is made, which simulates a human annotation oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import URLError

import certifi
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import FakeData, GTSRB


NUM_CLASSES = 43


@dataclass(frozen=True)
class RunConfig:
    data_dir: str
    output_dir: str
    strategies: tuple[str, ...]
    rounds: int
    query_size: int
    initial_per_class: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_fraction: float
    seed: int
    num_workers: int
    image_size: int
    max_train_samples: int | None
    max_test_samples: int | None
    smoke_test: bool


class TrafficSignCNN(nn.Module):
    """Compact CNN suitable for repeated retraining in active-learning rounds."""

    def __init__(self, image_size: int = 48, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        pooled_size = image_size // 8
        if pooled_size < 1:
            raise ValueError("image_size must be at least 8 pixels")
        self.features = nn.Sequential(
            self._block(3, 32),
            nn.MaxPool2d(2),
            self._block(32, 64),
            nn.MaxPool2d(2),
            self._block(64, 128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pooled_size * pooled_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_https_certificates() -> None:
    """Use certifi's CA store for Python builds without macOS root certificates."""
    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)


def label_of(dataset: Dataset, index: int) -> int:
    samples = getattr(dataset, "_samples", None)
    if samples is not None:
        return int(samples[index][1])
    item = dataset[index]
    return int(item[1])


def subset_indices(indices: Sequence[int], limit: int | None, seed: int) -> list[int]:
    values = list(indices)
    if limit is None or limit >= len(values):
        return values
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(values, size=limit, replace=False).tolist())


def make_datasets(
    config: RunConfig,
) -> tuple[Dataset, Dataset, Dataset, list[int], list[int]]:
    normalize = transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if config.smoke_test:
        train_dataset = FakeData(
            size=config.max_train_samples or 600,
            image_size=(3, config.image_size, config.image_size),
            num_classes=NUM_CLASSES,
            transform=train_transform,
            random_offset=config.seed,
        )
        pool_dataset = FakeData(
            size=config.max_train_samples or 600,
            image_size=(3, config.image_size, config.image_size),
            num_classes=NUM_CLASSES,
            transform=eval_transform,
            random_offset=config.seed,
        )
        test_dataset = FakeData(
            size=config.max_test_samples or 180,
            image_size=(3, config.image_size, config.image_size),
            num_classes=NUM_CLASSES,
            transform=eval_transform,
            random_offset=config.seed + 10_000,
        )
    else:
        configure_https_certificates()
        try:
            train_dataset = GTSRB(
                config.data_dir, split="train", download=True, transform=train_transform
            )
            pool_dataset = GTSRB(
                config.data_dir, split="train", download=False, transform=eval_transform
            )
            test_dataset = GTSRB(
                config.data_dir, split="test", download=True, transform=eval_transform
            )
        except URLError as error:
            raise RuntimeError(
                "Не удалось загрузить GTSRB по HTTPS. Проверьте подключение к сети и "
                "выполните 'pip install -r requirements.txt', чтобы установить certifi."
            ) from error

    train_indices = subset_indices(range(len(pool_dataset)), config.max_train_samples, config.seed)
    test_indices = subset_indices(range(len(test_dataset)), config.max_test_samples, config.seed + 1)
    return train_dataset, pool_dataset, test_dataset, train_indices, test_indices


def initial_pool_split(
    dataset: Dataset,
    indices: Sequence[int],
    initial_per_class: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {class_id: [] for class_id in range(NUM_CLASSES)}
    for index in indices:
        by_class[label_of(dataset, index)].append(index)

    validation: list[int] = []
    labeled: list[int] = []
    unlabeled: list[int] = []
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        if not class_indices:
            continue
        validation_count = min(
            max(1, int(round(len(class_indices) * validation_fraction))),
            max(0, len(class_indices) - 1),
        )
        class_validation = class_indices[:validation_count]
        remaining = class_indices[validation_count:]
        labeled_count = min(initial_per_class, len(remaining))
        validation.extend(class_validation)
        labeled.extend(remaining[:labeled_count])
        unlabeled.extend(remaining[labeled_count:])

    if not labeled:
        raise RuntimeError("The initial labeled set is empty; increase max_train_samples.")
    rng.shuffle(validation)
    rng.shuffle(labeled)
    rng.shuffle(unlabeled)
    return labeled, unlabeled, validation


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def train_model(
    dataset: Dataset,
    labeled_indices: Sequence[int],
    config: RunConfig,
    device: torch.device,
    round_seed: int,
) -> nn.Module:
    seed_everything(round_seed)
    model = TrafficSignCNN(image_size=config.image_size).to(device)
    loader = make_loader(
        dataset, labeled_indices, config.batch_size, config.num_workers, True, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(config.epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    config: RunConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if not indices:
        return np.empty((0, NUM_CLASSES), dtype=np.float32), np.empty(0, dtype=np.int64)
    loader = make_loader(dataset, indices, config.batch_size, config.num_workers, False, device)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    config: RunConfig,
    device: torch.device,
) -> tuple[float, float]:
    probabilities, targets = predict(model, dataset, indices, config, device)
    predicted = probabilities.argmax(axis=1)
    return (
        float(accuracy_score(targets, predicted)),
        float(f1_score(targets, predicted, average="macro", zero_division=0)),
    )


def uncertainty_scores(probabilities: np.ndarray, strategy: str) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    if strategy == "entropy":
        return -(clipped * np.log(clipped)).sum(axis=1)
    if strategy == "margin":
        two_best = np.partition(probabilities, -2, axis=1)[:, -2:]
        return -(two_best.max(axis=1) - two_best.min(axis=1))
    if strategy == "least_confidence":
        return 1.0 - probabilities.max(axis=1)
    raise ValueError(f"Unknown uncertainty strategy: {strategy}")


def query_indices(
    model: nn.Module,
    dataset: Dataset,
    unlabeled_indices: Sequence[int],
    strategy: str,
    query_size: int,
    config: RunConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> list[int]:
    count = min(query_size, len(unlabeled_indices))
    if count == 0:
        return []
    if strategy == "random":
        positions = rng.choice(len(unlabeled_indices), size=count, replace=False)
    else:
        probabilities, _ = predict(model, dataset, unlabeled_indices, config, device)
        scores = uncertainty_scores(probabilities, strategy)
        positions = np.argpartition(scores, -count)[-count:]
        positions = positions[np.argsort(scores[positions])[::-1]]
    return [int(unlabeled_indices[position]) for position in positions]


def run_strategy(
    strategy: str,
    train_dataset: Dataset,
    pool_dataset: Dataset,
    test_dataset: Dataset,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    config: RunConfig,
    device: torch.device,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, int | str]]]:
    labeled, unlabeled, validation = initial_pool_split(
        pool_dataset,
        train_indices,
        config.initial_per_class,
        config.validation_fraction,
        config.seed,
    )
    rng = np.random.default_rng(config.seed + sum(ord(char) for char in strategy))
    rows: list[dict[str, float | int | str]] = []
    selected_rows: list[dict[str, int | str]] = []

    for round_id in range(config.rounds + 1):
        model = train_model(train_dataset, labeled, config, device, config.seed + round_id)
        val_accuracy, val_f1 = evaluate(model, pool_dataset, validation, config, device)
        test_accuracy, test_f1 = evaluate(model, test_dataset, test_indices, config, device)
        rows.append(
            {
                "strategy": strategy,
                "round": round_id,
                "labeled_count": len(labeled),
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": val_f1,
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_f1,
            }
        )
        print(
            f"[{strategy}] round={round_id} labeled={len(labeled)} "
            f"test_accuracy={test_accuracy:.4f} macro_f1={test_f1:.4f}"
        )
        if round_id == config.rounds or not unlabeled:
            break
        queried = query_indices(
            model, pool_dataset, unlabeled, strategy, config.query_size, config, device, rng
        )
        queried_set = set(queried)
        for index in queried:
            selected_rows.append(
                {
                    "strategy": strategy,
                    "round": round_id + 1,
                    "dataset_index": index,
                    "oracle_class": label_of(pool_dataset, index),
                }
            )
        labeled.extend(queried)
        unlabeled = [index for index in unlabeled if index not in queried_set]
    return rows, selected_rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(rows: Sequence[dict], output_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.0))
    for strategy in sorted({str(row["strategy"]) for row in rows}):
        selected = [row for row in rows if row["strategy"] == strategy]
        plt.plot(
            [int(row["labeled_count"]) for row in selected],
            [float(row["test_accuracy"]) for row in selected],
            marker="o",
            linewidth=2,
            label=strategy,
        )
    plt.xlabel("Количество размеченных изображений")
    plt.ylabel("Accuracy на тестовой выборке")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--strategies", nargs="+", default=["entropy", "margin", "random"],
        choices=["entropy", "margin", "least_confidence", "random"],
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--query-size", type=int, default=600)
    parser.add_argument("--initial-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.rounds < 0 or args.query_size <= 0 or args.initial_per_class <= 0:
        parser.error("rounds must be non-negative; query-size and initial-per-class must be positive")
    return RunConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        strategies=tuple(args.strategies),
        rounds=args.rounds,
        query_size=args.query_size,
        initial_per_class=args.initial_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        image_size=args.image_size,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        smoke_test=args.smoke_test,
    )


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    device = choose_device()
    print(f"device={device}")
    train_dataset, pool_dataset, test_dataset, train_indices, test_indices = make_datasets(config)

    all_metrics: list[dict] = []
    all_selected: list[dict] = []
    for strategy in config.strategies:
        metrics, selected = run_strategy(
            strategy,
            train_dataset,
            pool_dataset,
            test_dataset,
            train_indices,
            test_indices,
            config,
            device,
        )
        all_metrics.extend(metrics)
        all_selected.extend(selected)

    write_csv(output_dir / "metrics.csv", all_metrics)
    write_csv(output_dir / "selected_samples.csv", all_selected)
    plot_learning_curves(all_metrics, output_dir / "learning_curve.png")
    print(f"results={output_dir.resolve()}")


if __name__ == "__main__":
    main()
