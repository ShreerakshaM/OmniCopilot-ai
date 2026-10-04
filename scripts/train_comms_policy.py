"""Phase 5 (Task 5.3) — train a PPO communication policy on CommunicationEnv.

Uses stable-baselines3 PPO. SB3 is a deliberate choice over a hand-rolled PPO: a
battle-tested implementation removes the ambiguity of "did it fail to converge because of
the problem, or because of a bug in my PPO?" — which matters for an honest Kill-Gate-C
verdict. The environment, reward, and baselines are ours; PPO is library-provided.

Pipeline:
  1. Train PPO on CommunicationEnv (sequential, budget-coupled).
  2. Log the learning curve (episode return over timesteps).
  3. Evaluate the trained policy vs. baselines (random, always-send, never-send) on the
     SAME env, reporting mean return, mean delta_mAP, and bandwidth spent.

Kill Gate C: if PPO return does not rise above the random baseline, STOP and report the
negative result honestly (fallback: imitation warm-start, then bandit — see spec).

Usage (Kaggle, GPU):
    pip install -q gymnasium stable-baselines3 torch
    python scripts/train_comms_policy.py \
        --data-root /kaggle/input/.../opv2v-2/test --module-dir build/cpp \
        --timesteps 50000 --out /kaggle/working/results/rl_policy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if _PY_ROOT.is_dir() and str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))


def make_env(data_root: str, module_dir: str | None, seed: int):
    from omnicopilot.rl.environment import CommunicationEnv  # noqa: PLC0415
    return CommunicationEnv(data_root=data_root, module_dir=module_dir, seed=seed)


def eval_policy(env, choose, episodes: int, rng) -> dict:
    """Run `episodes` with an action-chooser; return mean return / delta / bandwidth."""
    returns, deltas, spent = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_r, ep_d, done = 0.0, [], False
        start_budget = env._episode_budget
        while not done:
            a = choose(obs)
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            ep_d.append(info["delta_map"])
            done = term or trunc
        returns.append(ep_r)
        deltas.append(float(np.mean(ep_d)) if ep_d else 0.0)
        spent.append(start_budget - env._budget_left)
    return {"mean_return": round(float(np.mean(returns)), 4),
            "mean_delta_map": round(float(np.mean(deltas)), 4),
            "mean_bytes_spent": round(float(np.mean(spent)), 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--module-dir", type=str, default=None)
    p.add_argument("--timesteps", type=int, default=50000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from stable_baselines3 import PPO  # noqa: PLC0415

    env = make_env(args.data_root, args.module_dir, args.seed)
    rng = np.random.default_rng(args.seed)
    adim = env.action_space.shape[0]

    # ── Baselines (evaluated on the same env, before training) ──
    baselines = {
        "random": eval_policy(env, lambda o: rng.random(adim).astype(np.float32),
                              args.eval_episodes, rng),
        "always_send": eval_policy(env, lambda o: np.ones(adim, np.float32),
                                   args.eval_episodes, rng),
        "never_send": eval_policy(env, lambda o: np.zeros(adim, np.float32),
                                  args.eval_episodes, rng),
    }
    print("Baselines:")
    for k, v in baselines.items():
        print(f"  {k:12} {v}")

    # ── Train PPO ──
    print(f"\nTraining PPO for {args.timesteps} timesteps...")
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, device="cpu",
                learning_rate=3e-4, n_steps=512, batch_size=128, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    model.learn(total_timesteps=args.timesteps)

    # ── Evaluate trained policy ──
    def ppo_choose(o):
        a, _ = model.predict(o, deterministic=True)
        return a
    learned = eval_policy(env, ppo_choose, args.eval_episodes, rng)
    print(f"\nLearned (PPO): {learned}")

    # ── Kill-Gate-C verdict ──
    beat_random = learned["mean_return"] > baselines["random"]["mean_return"]
    verdict = ("PASS: PPO beat the random baseline"
               if beat_random else
               "KILL GATE C: PPO did NOT beat random — fall back to imitation/bandit")
    print(f"\n{verdict}")

    summary = {"baselines": baselines, "learned": learned,
               "beat_random": beat_random, "timesteps": args.timesteps}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "rl_summary.json").write_text(json.dumps(summary, indent=2))
        model.save(str(args.out / "ppo_comms_policy"))
        print(f"Wrote -> {args.out}")


if __name__ == "__main__":
    main()
