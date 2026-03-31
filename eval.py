import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _safe_mean(values: List[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    jsonl_path = run_dir / "per_prompt_metrics.jsonl"
    if jsonl_path.exists():
        rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    # Fallback: collect prompt-level results.json files.
    rows = []
    for p in sorted(run_dir.glob("*/results.json")):
        with open(p, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def _resolve_run_dir(path: Path) -> Path:
    # If path itself contains per_prompt_metrics.jsonl or results files, use it directly.
    if (path / "per_prompt_metrics.jsonl").exists() or list(path.glob("*/results.json")):
        return path

    # Otherwise treat path as parent (e.g., benchmark_ir_outputs_evolve) and pick latest timestamp dir.
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run directories found under: {path}")
    return sorted(subdirs)[-1]


def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Final-evolve stats (as saved in row-level summary)
    final_best_rewards = [float(r.get("best_reward", float("nan"))) for r in rows]
    final_means = [float(r.get("mean_reward", float("nan"))) for r in rows]

    # Base CFG stats = first loop (loop index 0 in loop_stats list)
    cfg_best_rewards = []
    cfg_means = []
    missing_loop_stats = 0

    for r in rows:
        loop_stats = r.get("loop_stats", [])
        if not loop_stats:
            missing_loop_stats += 1
            continue

        first = loop_stats[0]
        # "best reward" for loop 1 = max reward in that loop
        cfg_best_rewards.append(float(first.get("max", float("nan"))))
        # "mean of means" for loop 1 = mean reward in that loop
        cfg_means.append(float(first.get("mean", float("nan"))))

    return {
        "num_prompts": len(rows),
        "num_prompts_with_loop_stats": len(cfg_means),
        "num_prompts_missing_loop_stats": missing_loop_stats,
        "final_evolve": {
            "mean_best_reward": _safe_mean(final_best_rewards),
            "mean_of_means": _safe_mean(final_means),
        },
        "base_cfg_first_loop": {
            "mean_best_reward": _safe_mean(cfg_best_rewards),
            "mean_of_means": _safe_mean(cfg_means),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate evolve benchmark outputs (final vs base-CFG first loop)."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="benchmark_ir_outputs_evolve",
        help="Path to a specific run timestamp dir, or parent dir containing run dirs.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save computed stats as JSON.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(Path(args.run_dir))
    rows = _load_rows(run_dir)

    if not rows:
        raise RuntimeError(f"No prompt rows found in: {run_dir}")

    stats = compute_stats(rows)

    print(f"Run directory: {run_dir}")
    print(f"Num prompts: {stats['num_prompts']}")
    print("")
    print("Final evolve stats")
    print(f"  Mean of best rewards: {stats['final_evolve']['mean_best_reward']:.6f}")
    print(f"  Mean of means:        {stats['final_evolve']['mean_of_means']:.6f}")
    print("")
    print("Base CFG (first loop) stats")
    print(f"  Mean of best rewards: {stats['base_cfg_first_loop']['mean_best_reward']:.6f}")
    print(f"  Mean of means:        {stats['base_cfg_first_loop']['mean_of_means']:.6f}")
    print("")
    print(f"Prompts missing loop_stats: {stats['num_prompts_missing_loop_stats']}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats JSON to: {out_path}")


if __name__ == "__main__":
    main()