"""
Train and compare dense, MoE, and DeepSeek-MoE minGPT models on a small
synthetic next-token benchmark.
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
    C.system.work_dir = "./out/moe_bench"

    C.data = MixedTaskDataset.get_default_config()

    C.model = GPT.get_default_config()
    C.model.model_type = None
    C.model.n_layer = 4
    C.model.n_head = 4
    C.model.n_embd = 64
    C.model.ffn_type = "mlp"
    C.model.moe_num_experts = 4
    C.model.moe_top_k = 2
    C.model.moe_num_shared_experts = 1
    C.model.moe_expert_hidden_mult = 4
    C.model.moe_aux_loss_weight = 1e-2

    C.train = CN()
    C.train.device = "auto"
    C.train.batch_size = 64
    C.train.num_workers = 0
    C.train.max_iters = 1500
    C.train.eval_interval = 150
    C.train.eval_batches = 20
    C.train.eval_examples = 256
    C.train.learning_rate = 3e-4
    C.train.betas = (0.9, 0.95)
    C.train.weight_decay = 0.1
    C.train.grad_norm_clip = 1.0

    C.experiment = CN()
    C.experiment.variants = ["mlp", "moe", "deepseek_moe"]

    return C


class MixedTaskDataset(Dataset):
    """
    Four fixed-length next-token tasks mixed together:
    - copy:    C12345|12345
    - reverse: R12345|54321
    - sort:    S41325|12345
    - add:     A123456|9750   (123 + 456 = 579, output reversed)
    """

    TASK_TO_INDEX = {
        "copy": 0,
        "reverse": 1,
        "sort": 2,
        "add": 3,
    }
    INDEX_TO_TASK = {v: k for k, v in TASK_TO_INDEX.items()}

    @staticmethod
    def get_default_config():
        C = CN()
        C.seq_digits = 5
        C.add_digits = 3
        C.train_size = 12000
        C.val_size = 2000
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.length = config.train_size if split == "train" else config.val_size
        self.vocab = list("0123456789ACRS|")
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.examples = self._build_examples()

    def get_vocab_size(self):
        return len(self.vocab)

    def get_block_size(self):
        return len(self.examples[0]["render"]) - 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        render = example["render"]
        output_start = example["output_start"]

        dix = [self.stoi[ch] for ch in render]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        y[: output_start - 1] = -1

        prompt_len = torch.tensor(output_start, dtype=torch.long)
        output_len = torch.tensor(len(render) - output_start, dtype=torch.long)
        task_idx = torch.tensor(self.TASK_TO_INDEX[example["task"]], dtype=torch.long)
        return x, y, prompt_len, output_len, task_idx

    def _build_examples(self):
        rng = random.Random(1234 if self.split == "train" else 4321)
        examples = []
        for _ in range(self.length):
            task = rng.choice(list(self.TASK_TO_INDEX))
            examples.append(self._sample_example(task, rng))
        return examples

    def _sample_example(self, task, rng):
        if task == "copy":
            digits = self._rand_digits(rng, self.config.seq_digits)
            render = "C" + digits + "|" + digits
            output_start = 1 + self.config.seq_digits + 1
        elif task == "reverse":
            digits = self._rand_digits(rng, self.config.seq_digits)
            render = "R" + digits + "|" + digits[::-1]
            output_start = 1 + self.config.seq_digits + 1
        elif task == "sort":
            digits = self._rand_digits(rng, self.config.seq_digits)
            render = "S" + digits + "|" + "".join(sorted(digits))
            output_start = 1 + self.config.seq_digits + 1
        elif task == "add":
            nd = 10 ** self.config.add_digits
            a = rng.randrange(nd)
            b = rng.randrange(nd)
            a_str = f"%0{self.config.add_digits}d" % a
            b_str = f"%0{self.config.add_digits}d" % b
            c_str = (f"%0{self.config.add_digits + 1}d" % (a + b))[::-1]
            render = "A" + a_str + b_str + "|" + c_str
            output_start = 1 + self.config.add_digits + self.config.add_digits + 1
        else:
            raise ValueError(f"Unknown task {task}")

        return {
            "task": task,
            "render": render,
            "output_start": output_start,
        }

    def _rand_digits(self, rng, n):
        return "".join(str(rng.randrange(10)) for _ in range(n))


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
            x, y = [t.to(device) for t in batch[:2]]
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def evaluate_exact_match(model, dataset, device, max_examples):
    model.eval()
    totals = {task: 0 for task in dataset.TASK_TO_INDEX}
    correct = {task: 0 for task in dataset.TASK_TO_INDEX}

    count = min(len(dataset), max_examples if max_examples is not None else len(dataset))
    with torch.no_grad():
        for idx in range(count):
            x, y, prompt_len, output_len, task_idx = dataset[idx]
            prompt_len = int(prompt_len.item())
            output_len = int(output_len.item())
            task_name = dataset.INDEX_TO_TASK[int(task_idx.item())]

            prompt = x[:prompt_len].unsqueeze(0).to(device)
            generated = model.generate(prompt, max_new_tokens=output_len, do_sample=False)
            pred = generated[0, prompt_len:prompt_len + output_len].cpu()
            target = y[prompt_len - 1:prompt_len - 1 + output_len]

            totals[task_name] += 1
            if torch.equal(pred, target):
                correct[task_name] += 1

    model.train()

    metrics = {}
    total_correct = 0
    total_seen = 0
    for task_name in dataset.TASK_TO_INDEX:
        task_total = totals[task_name]
        task_correct = correct[task_name]
        total_correct += task_correct
        total_seen += task_total
        metrics[task_name] = task_correct / max(task_total, 1)
    metrics["overall"] = total_correct / max(total_seen, 1)
    return metrics


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

        x, y = [t.to(device) for t in batch[:2]]
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_norm_clip)
        optimizer.step()

        iter_num += 1

        if iter_num == 1 or iter_num % config.train.eval_interval == 0 or iter_num == config.train.max_iters:
            val_loss = estimate_loss(model, val_loader, device, config.train.eval_batches)
            exact = evaluate_exact_match(model, val_dataset, device, config.train.eval_examples)
            moe_aux = None if model.last_moe_aux_loss is None else float(model.last_moe_aux_loss.detach().cpu().item())

            metrics = {
                "iter": iter_num,
                "train_loss": float(loss.detach().cpu().item()),
                "val_loss": float(val_loss),
                "val_exact": exact,
                "moe_aux_loss": moe_aux,
            }
            history.append(copy.deepcopy(metrics))
            print(
                f"[{variant_name}] iter {iter_num}: "
                f"train_loss={metrics['train_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"val_exact={metrics['val_exact']['overall']:.4f} "
                f"moe_aux={metrics['moe_aux_loss']}"
            )

            rank_key = (metrics["val_exact"]["overall"], -metrics["val_loss"])
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_metrics = metrics
                best_snapshot = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    elapsed = time.time() - start_time

    variant_dir = os.path.join(config.system.work_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)
    best_metrics["elapsed_seconds"] = elapsed
    best_metrics["num_parameters"] = sum(p.numel() for p in model.parameters())
    best_metrics["history"] = history
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

    train_dataset = MixedTaskDataset(config.data, split="train")
    val_dataset = MixedTaskDataset(config.data, split="val")

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
            f"val_exact={metrics['val_exact']['overall']:.4f}, "
            f"params={metrics['num_parameters']}"
        )
