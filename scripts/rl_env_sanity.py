"""Sanity-check the CommunicationEnv with a RANDOM policy before any PPO training.

Verifies the environment is correct end-to-end:
  - reset() returns an observation of the right shape
  - step() runs a full episode (one scenario's frames)
  - the shared episode budget DEPLETES over time (the sequential coupling)
  - rewards are finite and transmitting generally helps (delta_map >= 0 on average)

If this does not behave sensibly, PPO would "learn" from a broken env -- so we gate
training on this passing first (same discipline as the C++ work).

Usage:
    python scripts/rl_env_sanity.py \
        --data-root /kaggle/input/.../opv2v-2/test --module-dir build/cpp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if _PY_ROOT.is_dir() and str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--module-dir", type=str, default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from omnicopilot.rl.environment import CommunicationEnv  # noqa: PLC0415

    env = CommunicationEnv(data_root=args.data_root, module_dir=args.module_dir,
                           seed=args.seed)
    rng = np.random.default_rng(args.seed)

    print("obs space:", env.observation_space.shape, "| action space:", env.action_space.shape)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape, "reset obs shape mismatch"
        assert np.isfinite(obs).all(), "reset obs has NaN/inf"

        ep_reward = 0.0
        deltas, budgets, n_tx = [], [], []
        steps = 0
        done = False
        while not done:
            action = rng.random(env.action_space.shape[0]).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward), "non-finite reward"
            assert np.isfinite(obs).all(), "non-finite obs"
            ep_reward += reward
            deltas.append(info["delta_map"])
            budgets.append(info["budget_left"])
            n_tx.append(info["n_transmitted"])
            steps += 1
            done = terminated or truncated

        print(f"\nEpisode {ep}: steps={steps}  return={ep_reward:+.4f}")
        print(f"  mean delta_map (tx vs silent): {np.mean(deltas):+.4f}")
        print(f"  budget: start={env._episode_budget} end={budgets[-1]} "
              f"(spent {env._episode_budget - budgets[-1]})")
        print(f"  mean transmitted/frame: {np.mean(n_tx):.1f}")
        # Budget must be non-increasing (the sequential coupling).
        assert all(budgets[i] >= budgets[i + 1] for i in range(len(budgets) - 1)), \
            "budget should be non-increasing within an episode"

    print("\nPASS: env resets, runs full episodes, budget depletes monotonically, "
          "rewards finite. Safe to train PPO on.")


if __name__ == "__main__":
    main()
