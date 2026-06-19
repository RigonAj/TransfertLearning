"""
UNN Policy — PPO operating in Cartesian latent space (velocity-based).

Action pipeline:
    PPO predicts (vx, vy) normalized in [-1, 1]          (end-effector velocity direction)
    → v_ee         = latent_action * v_max_ee             (m/s)
    → omega_joints = J^T (J J^T + λ²I)^{-1} v_ee        (rad/s, via ik_velocity)
    → joint_action = omega_joints / omega_max             (normalized, clipped to [-1,1])
    → env step:    Δθ = joint_action * omega_max * dt     (rad)

Load() works without any environment (no DummyEnv, no Gymnasium dependency at inference).
"""

import numpy as np
import torch
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from unn.bases_unn import CartesianStateEncoder, ik_velocity


# ---------------------------------------------------------------------------
# Lightweight normalizer — replaces VecNormalize at inference
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """
    Applies the same obs normalization as VecNormalize but with no env dependency.
    """
    def __init__(self, mean: np.ndarray, var: np.ndarray,
                 clip_obs: float = 10.0, epsilon: float = 1e-8):
        self.mean     = mean
        self.var      = var
        self.clip_obs = clip_obs
        self.epsilon  = epsilon

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        normed = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)

    def save(self, path: Path):
        torch.save({"mean": self.mean, "var": self.var,
                    "clip_obs": self.clip_obs, "epsilon": self.epsilon}, path)

    @classmethod
    def load(cls, path: Path) -> "RunningNormalizer":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["mean"], d["var"], d.get("clip_obs", 10.0), d.get("epsilon", 1e-8))

    @classmethod
    def from_vec_normalize(cls, vec_norm: VecNormalize) -> "RunningNormalizer":
        return cls(
            mean=vec_norm.obs_rms.mean.copy(),
            var=vec_norm.obs_rms.var.copy(),
            clip_obs=float(vec_norm.clip_obs),
            epsilon=float(vec_norm.epsilon)
        )


# ---------------------------------------------------------------------------
# UNNPolicy
# ---------------------------------------------------------------------------

class UNNPolicy:
    """
    Wraps a PPO policy that operates in a 6D Cartesian latent space.

    The policy predicts normalized end-effector velocities (vx, vy) in [-1,1].
    These are converted to joint velocity commands via the analytic Jacobian
    pseudo-inverse (ik_velocity), then normalized for the environment.

    Parameters
    ----------
    v_max_ee : float
        Maximum end-effector speed in m/s.
        Typical value: omega_max * max_reach (e.g. 2.0 * 3.0 = 6.0 m/s).
        The PPO output is scaled by this before IK: v_ee = latent_action * v_max_ee.
    """

    def __init__(
        self,
        encoder: CartesianStateEncoder,
        ppo_policy: PPO,
        vec_normalize,           # VecNormalize (training) | RunningNormalizer (inference) | None
        arm_obs_size: int,
        n_joints: int,
        link_lengths: list = None,
        omega_max: float = None,
        v_max_ee: float = None,  # NEW — max EE speed (m/s)
        dt: float = None,        # kept for metadata / backward compat, not used in IK
        device: str = "cpu"
    ):
        self.encoder      = encoder
        self.ppo_policy   = ppo_policy
        self.vec_normalize = vec_normalize
        self.arm_obs_size = arm_obs_size
        self.n_joints     = n_joints
        self.link_lengths = None if link_lengths is None else list(link_lengths)
        self.omega_max    = omega_max
        self.v_max_ee     = v_max_ee
        self.dt           = dt   # stored for reference, not used in predict()
        self.device       = device

        self.ppo_policy.policy.eval()

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, raw_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict a joint velocity command from a raw environment observation.

        Steps:
          1. Encode raw_obs → 6D Cartesian latent state
          2. Normalize latent state
          3. PPO → normalized EE velocity (vx, vy) in [-1, 1]
          4. Scale: v_ee = latent_action * v_max_ee   (m/s)
          5. IK:   omega_joints = J^+ v_ee            (rad/s)
          6. Normalize: joint_action = omega_joints / omega_max

        Returns
        -------
        joint_action : np.ndarray, shape (n_joints,), clipped to [-1, 1]
        """
        # 1) Encode
        latent_state = self.encoder.encode(raw_obs, self.arm_obs_size)

        # 2) Normalize
        if self.vec_normalize is not None:
            if isinstance(self.vec_normalize, RunningNormalizer):
                latent_state = self.vec_normalize.normalize(latent_state)
            else:
                latent_state = self.vec_normalize.normalize_obs(
                    latent_state.reshape(1, -1)
                )[0]

        # 3) PPO → normalized EE velocity
        latent_action, _ = self.ppo_policy.predict(latent_state, deterministic=deterministic)
        latent_action = np.asarray(latent_action, dtype=np.float32).reshape(-1)
        if latent_action.shape[0] != 2:
            raise ValueError(f"latent_action must be (2,), got {latent_action.shape}")

        # 4) Check required parameters
        if self.link_lengths is None or self.omega_max is None or self.v_max_ee is None:
            raise RuntimeError(
                "UNNPolicy requires `link_lengths`, `omega_max`, and `v_max_ee` "
                "to perform velocity-based IK conversion."
            )

        # 5) Extract current joint angles in radians (raw_obs stores θ/π)
        joint_angles_rad = (
            np.asarray(raw_obs[:self.n_joints], dtype=np.float32).reshape(-1) * np.pi
        )

        # 6) Scale latent action → EE velocity (m/s)
        v_ee = latent_action * float(self.v_max_ee)

        # 7) IK: EE velocity → joint velocities (rad/s)
        omega_joints = ik_velocity(
            joint_angles_rad, v_ee,
            np.array(self.link_lengths, dtype=np.float32)
        )

        # 8) Normalize to [-1, 1] for the environment
        joint_action = (omega_joints / float(self.omega_max)).astype(np.float32)
        joint_action = np.clip(joint_action, -1.0, 1.0)

        return joint_action.reshape(self.n_joints,)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(self, save_dir: str, name: str = "unn_policy"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # PPO weights
        self.ppo_policy.save(save_path / f"{name}_ppo.zip")

        # Normalizer stats
        if self.vec_normalize is not None:
            if isinstance(self.vec_normalize, RunningNormalizer):
                self.vec_normalize.save(save_path / f"{name}_norm_stats.pt")
            else:
                rn = RunningNormalizer.from_vec_normalize(self.vec_normalize)
                rn.save(save_path / f"{name}_norm_stats.pt")
                self.vec_normalize.save(save_path / f"{name}_vecnorm.pkl")

        # Metadata — includes v_max_ee
        meta = {
            "arm_obs_size": self.arm_obs_size,
            "n_joints":     self.n_joints,
        }
        if self.link_lengths is not None:
            meta["link_lengths"] = list(self.link_lengths)
        if self.omega_max is not None:
            meta["omega_max"] = float(self.omega_max)
        if self.v_max_ee is not None:
            meta["v_max_ee"] = float(self.v_max_ee)
        if self.dt is not None:
            meta["dt"] = float(self.dt)

        torch.save(meta, save_path / f"{name}_meta.pt")
        print(f"[UNN] Policy saved → {save_path}")

    # -----------------------------------------------------------------------
    # Load — no environment required
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, load_dir: str, name: str = "unn_policy",
             encoder: CartesianStateEncoder = None, device: str = "cpu") -> "UNNPolicy":
        """
        Load UNNPolicy for inference.
        Does NOT require any Gymnasium environment or SB3 VecEnv.
        """
        load_path = Path(load_dir)

        # 1) Metadata
        meta = torch.load(load_path / f"{name}_meta.pt", map_location=device, weights_only=False)
        n_joints     = meta["n_joints"]
        arm_obs_size = meta["arm_obs_size"]
        link_lengths = meta.get("link_lengths", None)
        omega_max    = meta.get("omega_max", None)
        v_max_ee     = meta.get("v_max_ee", None)
        dt           = meta.get("dt", None)

        # 2) PPO
        ppo = PPO.load(load_path / f"{name}_ppo.zip", device=device)

        # 3) Normalizer
        normalizer = None
        stats_path = load_path / f"{name}_norm_stats.pt"
        pkl_path   = load_path / f"{name}_vecnorm.pkl"

        if stats_path.exists():
            normalizer = RunningNormalizer.load(stats_path)
            print(f"[UNN] Loaded normalizer stats from {stats_path.name}")
        elif pkl_path.exists():
            print(f"[UNN] {stats_path.name} not found — extracting stats from {pkl_path.name}")
            normalizer = _load_vecnorm_stats(pkl_path)

        # 4) Encoder
        if encoder is None:
            encoder = CartesianStateEncoder()

        print(f"[UNN] Policy '{name}' loaded from {load_path}")

        return cls(
            encoder=encoder,
            ppo_policy=ppo,
            vec_normalize=normalizer,
            arm_obs_size=arm_obs_size,
            n_joints=n_joints,
            link_lengths=link_lengths,
            omega_max=omega_max,
            v_max_ee=v_max_ee,
            dt=dt,
            device=device
        )


# ---------------------------------------------------------------------------
# Internal helper — legacy .pkl migration
# ---------------------------------------------------------------------------

def _load_vecnorm_stats(pkl_path: Path) -> RunningNormalizer:
    """
    Load a VecNormalize .pkl, extract running stats, return a RunningNormalizer.
    """
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _MinimalLatentEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-10.0, high=10.0, shape=(6,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    dummy_venv = DummyVecEnv([lambda: _MinimalLatentEnv()])
    vec_norm   = VecNormalize.load(str(pkl_path), venv=dummy_venv)
    vec_norm.training    = False
    vec_norm.norm_reward = False

    return RunningNormalizer.from_vec_normalize(vec_norm)
