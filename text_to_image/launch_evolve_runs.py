import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from diffusers import DDIMScheduler

from evolve_diffusers import BaseSDXL
from evolve_diffusers.steer_pipeline import iterative_sample_with_stein


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_prompt_data(prompt_path: str, max_prompts: Optional[int] = None) -> List[Dict[str, Any]]:
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif prompt_path.endswith(".jsonl"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError("prompt_path must end with .json or .jsonl")

    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON list or JSONL rows")

    for item in data:
        if "prompt" not in item:
            if "text" in item:
                item["prompt"] = item["text"]
            else:
                raise ValueError("Each prompt item must contain 'prompt' or 'text'")

    if max_prompts is not None:
        data = data[:max_prompts]

    return data


def make_output_dir(base_output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(base_output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
        }

    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "max": float(x.max().item()),
        "min": float(x.min().item()),
    }


def save_final_images(images: List[Any], prompt_output_dir: Path) -> int:
    """Save only the final image pool for a prompt."""
    sample_dir = prompt_output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for image_idx, image in enumerate(images):
        image.save(sample_dir / f"{image_idx:05d}.png")
    return len(images)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    prompt_data = load_prompt_data(args.prompt_path, max_prompts=args.max_prompts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = BaseSDXL.from_pretrained(args.model_name, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    output_dir = make_output_dir(args.output_dir)

    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    per_prompt_jsonl = output_dir / "per_prompt_metrics.jsonl"
    per_prompt_rows: List[Dict[str, Any]] = []

    for prompt_idx, item in enumerate(prompt_data):
        prompt_text = item["prompt"]
        item_id = item.get("id", f"prompt_{prompt_idx:05d}")

        prompt_output_dir = output_dir / f"{prompt_idx:05d}"
        prompt_output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(args.seed + prompt_idx)

        start_time = datetime.now()
        loop_out = iterative_sample_with_stein(
            model=pipe,
            prompt=prompt_text,
            num_loops=args.num_loops,
            num_particles=args.num_particles,
            steer_start_timestep=args.steer_start_timestep,
            steer_end_timestep=args.steer_end_timestep,
            stein_step_size=args.stein_step_size,
            stein_num_steps=args.stein_num_steps,
            stein_weight_temperature=args.stein_weight_temperature,
            stein_langevin_lambda=args.stein_langevin_lambda,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            guidance_reward_fn=args.guidance_reward_fn,
            metric_to_chase=args.metric_to_chase,
            early_stop_epsilon=args.early_stop_epsilon,
        )
        elapsed_seconds = (datetime.now() - start_time).total_seconds()

        completed_loops = loop_out.get("completed_loops", 0)
        if completed_loops == 0:
            final_rewards = torch.empty(0, dtype=torch.float32)
            loop_stats = []
            final_images: List[Any] = []
        else:
            final_rewards = loop_out["rewards"][-1].detach().cpu().float()
            loop_stats = []
            for loop_i, rewards_tensor in enumerate(loop_out["rewards"], start=1):
                loop_rewards = rewards_tensor.detach().cpu().float()
                stats = tensor_stats(loop_rewards)
                stats["loop"] = loop_i
                stats["num_particles"] = int(loop_rewards.numel())
                loop_stats.append(stats)

            # Final pool includes carried accepted images plus last-loop resampled/rejected images.
            final_images = [
                *list(loop_out.get("accepted_images_pool", []) or []),
                *list(loop_out.get("rejected_images_pool", []) or []),
            ]
            if not final_images:
                final_images = list(getattr(loop_out.get("results", [])[-1], "images", []) or [])

        if args.save_individual_images:
            saved_images_count = save_final_images(final_images, prompt_output_dir)
        else:
            saved_images_count = len(final_images)

        final_stats = tensor_stats(final_rewards)

        row = {
            "prompt_index": prompt_idx,
            "id": item_id,
            "prompt": prompt_text,
            "reward_name": args.guidance_reward_fn,
            "num_particles": int(final_rewards.numel()),
            "completed_loops": int(completed_loops),
            "early_stopped": bool(loop_out.get("early_stopped", False)),
            "early_stop_reason": loop_out.get("early_stop_reason"),
            "best_mean_reward": float(loop_out.get("best_mean_reward", float("nan"))),
            "best_reward": float(loop_out.get("best_reward", float("nan"))),
            "mean_reward": final_stats["mean"],
            "std_reward": final_stats["std"],
            "max_reward": final_stats["max"],
            "min_reward": final_stats["min"],
            "time_seconds": elapsed_seconds,
            "thresholds": [float(x) for x in loop_out.get("thresholds", [])],
            "loop_stats": loop_stats,
            "final_saved_images": int(saved_images_count),
        }

        per_prompt_rows.append(row)

        with open(per_prompt_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        with open(prompt_output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)
        with open(prompt_output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

        print(
            f"[{prompt_idx + 1}/{len(prompt_data)}] "
            f"id={item_id} "
            f"mean={row['mean_reward']:.4f} "
            f"std={row['std_reward']:.4f} "
            f"max={row['max_reward']:.4f} "
            f"min={row['min_reward']:.4f} "
            f"loops={row['completed_loops']} "
            f"images={row['final_saved_images']} "
            f"secs={row['time_seconds']:.2f}"
        )

    summary = {
        "num_prompts": len(per_prompt_rows),
        "reward_name": args.guidance_reward_fn,
        "avg_mean_reward": float(np.mean([r["mean_reward"] for r in per_prompt_rows])) if per_prompt_rows else float("nan"),
        "avg_std_reward": float(np.mean([r["std_reward"] for r in per_prompt_rows])) if per_prompt_rows else float("nan"),
        "avg_max_reward": float(np.mean([r["max_reward"] for r in per_prompt_rows])) if per_prompt_rows else float("nan"),
        "avg_min_reward": float(np.mean([r["min_reward"] for r in per_prompt_rows])) if per_prompt_rows else float("nan"),
        "avg_time_seconds": float(np.mean([r["time_seconds"] for r in per_prompt_rows])) if per_prompt_rows else float("nan"),
    }

    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved results to:", str(output_dir))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterative Stein steering on prompt sets")

    parser.add_argument("--prompt_path", type=str, default="prompt_files/benchmark_ir.json")
    parser.add_argument("--max_prompts", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="benchmark_ir_outputs_evolve")
    parser.add_argument("--save_individual_images", action="store_true", default=True)

    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_loops", type=int, default=4)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)

    parser.add_argument("--steer_start_timestep", type=int, default=400)
    parser.add_argument("--steer_end_timestep", type=int, default=150)
    parser.add_argument("--stein_step_size", type=float, default=1e-6)
    parser.add_argument("--stein_num_steps", type=int, default=10)
    parser.add_argument("--stein_weight_temperature", type=float, default=1.0)
    parser.add_argument("--stein_langevin_lambda", type=float, default=1.0)

    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument("--metric_to_chase", type=str, default=None)
    parser.add_argument("--early_stop_epsilon", type=float, default=1e-5)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
