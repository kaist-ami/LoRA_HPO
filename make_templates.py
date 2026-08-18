import argparse
import pandas as pd

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ALPHA_MULTIPLIERS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
DROPOUTS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
BATCH_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
LEARNING_RATES = [1e-6, 5e-6, 2e-5, 3e-5, 5e-5, 1e-4, 3e-4, 4e-4, 5e-4, 5e-3]

PROSE = (
    "* **Rank (r):** Controls adapter capacity by setting the low-rank dimension—higher r increases expressivity (and memory/compute) but raises overfitting risk. If you raise r, consider stronger regularization or a slightly lower learning rate.\n"
    "* **Alpha (scaling):** Scales the LoRA update; the effective update magnitude is \\~**alpha / r**, so setting alpha ≈ r keeps update strength stable. Larger alpha amplifies adaptation but can destabilize training if LR is high.\n"
    "* **Batch size:** Number of samples per optimizer step—larger batches give smoother gradients and typically permit a proportionally larger learning rate (linear-scaling rule) at the cost of more memory. Small batches may need gradient accumulation or a reduced LR.\n"
    "* **Learning rate:** Step size for adapter parameters—too high can diverge (especially with large alpha/r), too low slows convergence. Tune in conjunction with batch size and consider schedules (e.g., cosine) to balance speed and stability.\n"
    "* **Dropout (LoRA dropout):** Probability of dropping the adapter path to regularize training; higher dropout curbs overfitting, especially with large r or small datasets. With higher dropout you can often afford slightly larger alpha or LR without instability.\n\n\n[Hyperparameters selection]\n"
)

DEFAULT_OUTPUT = {
    "frozen": "data/hyperparameters.csv",
    "learnable": "data/hyperparameters_learnable_token.csv",
}


def build(mode):
    tail = "<TT>" if mode == "learnable" else ""
    rows = []
    for rank in RANKS:
        for alpha in [m * rank for m in ALPHA_MULTIPLIERS]:
            for dropout in DROPOUTS:
                for batch_size in BATCH_SIZES:
                    for lr in LEARNING_RATES:
                        string = (
                            f"rank={rank}, alpha={alpha}, dropout={dropout}, "
                            f"batch_size={batch_size}, learning_rate={lr}"
                        )
                        template = (
                            PROSE
                            + f"rank={rank}\n"
                            + f"alpha={alpha}\n"
                            + f"dropout={dropout}\n"
                            + f"batch_size={batch_size}\n"
                            + f"learning_rate={lr}{tail}"
                        )
                        rows.append(
                            {"string": string, "template": template, "gsm8k_acc": -1, "math_acc": -1}
                        )

    df = pd.DataFrame(rows, columns=["string", "template", "gsm8k_acc", "math_acc"])
    df["total_accuracy"] = df["gsm8k_acc"] + df["math_acc"]
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate the hyperparameter template table")
    parser.add_argument("--mode", default="frozen", choices=list(DEFAULT_OUTPUT.keys()))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or DEFAULT_OUTPUT[args.mode]
    df = build(args.mode)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} configurations -> {output}")


if __name__ == "__main__":
    main()
