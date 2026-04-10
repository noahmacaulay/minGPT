"""
Train and compare dense, MoE, and DeepSeek-MoE minGPT models on a small
TinyStories next-token benchmark.
"""

import copy
import json
import os
import random
import sys
import time

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from mingpt.model import GPT
from mingpt.utils import set_seed, setup_logging, CfgNode as CN


def get_config():
    C = CN()

    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/tinystories_moe_bench"

    C.data = TinyStoriesCharDataset.get_default_config()

    C.model = GPT.get_default_config()
    C.model.model_type = None
    C.model.n_layer = 4
    C.model.n_head = 4
    C.model.n_embd = 128
    C.model.ffn_type = "mlp"
    C.model.moe_num_experts = 4
    C.model.moe_top_k = 2
    C.model.moe_num_shared_experts = 1
    C.model.moe_expert_hidden_mult = 4
    C.model.moe_aux_loss_weight = 1e-2

    C.train = CN()
    C.train.device = "auto"
    C.train.batch_size = 32
    C.train.num_workers = 0
    C.train.max_iters = 2000
    C.train.eval_interval = 200
    C.train.eval_batches = 20
    C.train.learning_rate = 3e-4
    C.train.betas = (0.9, 0.95)
    C.train.weight_decay = 0.1
    C.train.grad_norm_clip = 1.0
    C.train.sample_tokens = 200

    C.experiment = CN()
    C.experiment.variants = ["mlp", "moe", "deepseek_moe"]

    return C


class TinyStoriesCharDataset(Dataset):
    """
    Character-level language modeling dataset built from a small TinyStories subset.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.dataset_name = "roneneldan/TinyStories"
        C.text_field = "text"
        C.train_split = "train"
        C.val_split = "validation"
        C.max_train_stories = 4000
        C.max_val_stories = 500
        C.block_size = 128
        C.story_separator = "\n\n<|endofstory|>\n\n"
        return C

    def __init__(self, config, split, stoi=None, itos=None):
        self.config = config
        self.split = split
        texts = load_tinystories_texts(config, split)
        self.data = config.story_separator.join(texts)

        if stoi is None or itos is None:
            chars = sorted(list(set(self.data)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
        else:
            self.stoi = stoi
            self.itos = itos

        self.unk_index = self.stoi.get(" ", 0)
        self.encoded = torch.tensor(
            [self.stoi.get(ch, self.unk_index) for ch in self.data],
            dtype=torch.long,
        )

    def get_vocab_size(self):
        return len(self.stoi)

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.encoded) - self.config.block_size

    def __getitem__(self, idx):
        chunk = self.encoded[idx: idx + self.config.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def encode(self, text):
        return [self.stoi.get(ch, self.unk_index) for ch in text]

    def decode(self, token_ids):
        return "".join(self.itos[int(i)] for i in token_ids)


def load_tinystories_texts(config, split):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for TinyStories. "
            "Install it in your .venv with `pip install datasets`."
        ) from exc

    if split == "train":
        split_name = f"{config.train_split}[:{config.max_train_stories}]"
    elif split == "val":
        split_name = f"{config.val_split}[:{config.max_val_stories}]"
    else:
        raise ValueError(f"Unknown split {split}")

    dataset = load_dataset(config.dataset_name, split=split_name)
    if config.text_field not in dataset.column_names:
        raise ValueError(
            f"Could not find text field `{config.text_field}` in dataset columns {dataset.column_names}"
        )

    texts = [record[config.text_field].strip() for record in dataset if record[config.text_field].strip()]
    if not texts:
        raise ValueError("TinyStories subset is empty after loading and filtering")
    return texts


def pick_device(device_config):
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def build_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )


def estimate_loss(model, loader, device, max_batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y = [t.to(device) for t in batch]
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def sample_completion(model, dataset, device, max_new_tokens):
    model.eval()
    start_idx = random.randint(0, len(dataset) - dataset.get_block_size() - 1)
    prompt_tokens = dataset.encoded[start_idx: start_idx + min(64, dataset.get_block_size() // 2)]
    prompt = prompt_tokens.unsqueeze(0).to(device)
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_k=20)
    model.train()
    return dataset.decode(generated[0].cpu().tolist())


def train_variant(config, variant_name, train_dataset, val_dataset):
    device = pick_device(config.train.device)

    model_config = GPT.get_default_config()
    model_config.merge_from_dict(copy.deepcopy(config.model.to_dict()))
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model_config.ffn_type = variant_name

    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizers(config.train)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=False,
    )

    best_snapshot = None
    best_metrics = None
    best_key = None
    history = []

    iter_num = 0
    data_iter = iter(train_loader)
    start_time = time.time()
    model.train()

    while iter_num < config.train.max_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x, y = [t.to(device) for t in batch]
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_norm_clip)
        optimizer.step()

        iter_num += 1

        if iter_num == 1 or iter_num % config.train.eval_interval == 0 or iter_num == config.train.max_iters:
            val_loss = estimate_loss(model, val_loader, device, config.train.eval_batches)
            moe_aux = None if model.last_moe_aux_loss is None else float(model.last_moe_aux_loss.detach().cpu().item())

            metrics = {
                "iter": iter_num,
                "train_loss": float(loss.detach().cpu().item()),
                "val_loss": float(val_loss),
                "moe_aux_loss": moe_aux,
            }
            history.append(copy.deepcopy(metrics))
            print(
                f"[{variant_name}] iter {iter_num}: "
                f"train_loss={metrics['train_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"moe_aux={metrics['moe_aux_loss']}"
            )

            rank_key = -metrics["val_loss"]
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_metrics = copy.deepcopy(metrics)
                best_snapshot = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    elapsed = time.time() - start_time

    best_metrics["elapsed_seconds"] = elapsed
    best_metrics["num_parameters"] = sum(p.numel() for p in model.parameters())
    best_metrics["history"] = history
    best_metrics["sample_completion"] = sample_completion(
        model,
        val_dataset,
        device,
        max_new_tokens=config.train.sample_tokens,
    )

    variant_dir = os.path.join(config.system.work_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)
    if best_snapshot is not None:
        torch.save(best_snapshot, os.path.join(variant_dir, "model.pt"))
    with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)

    return best_metrics


def maybe_plot_histories(summary, work_dir):
    if plt is None:
        print("matplotlib is not installed; skipping loss plots")
        return

    plot_specs = [
        ("train_loss", "Training Loss", "train_loss.png"),
        ("val_loss", "Validation Loss", "val_loss.png"),
    ]

    for metric_key, title, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant_name, metrics in summary.items():
            history = metrics.get("history", [])
            if not history:
                continue
            xs = [point["iter"] for point in history]
            ys = [point[metric_key] for point in history]
            ax.plot(xs, ys, marker="o", linewidth=2, label=variant_name)

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(work_dir, filename), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    train_dataset = TinyStoriesCharDataset(config.data, split="train")
    val_dataset = TinyStoriesCharDataset(
        config.data,
        split="val",
        stoi=train_dataset.stoi,
        itos=train_dataset.itos,
    )

    summary = {}
    for variant_name in config.experiment.variants:
        print(f"\n=== Training {variant_name} ===")
        summary[variant_name] = train_variant(config, variant_name, train_dataset, val_dataset)

    maybe_plot_histories(summary, config.system.work_dir)

    summary_path = os.path.join(config.system.work_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final Summary ===")
    for variant_name, metrics in summary.items():
        print(
            f"{variant_name}: "
            f"val_loss={metrics['val_loss']:.4f}, "
            f"params={metrics['num_parameters']}"
        )
