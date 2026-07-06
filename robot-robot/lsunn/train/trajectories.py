"""
Phase 1: Génération de trajectoires primitives de reaching via IK analytique.
Pas d'interaction avec Gym - génération cinématique directe.
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.kinematics_ik import ik_2dof, ik_3dof, ik_2dof_torch, ik_3dof_torch

# Dimensions
ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8

# Paramètres de génération
N_PAIRS = 100_000
SEQ_LEN = 30
DT = 1/40  # 40 Hz pour génération, indépendant du dt=0.05 des environnements RL
L1_2D = 1.5
L2_2D = 1.5
L1_3D = 1.0
L2_3D = 1.0
L3_3D = 1.0
THETA_TOTAL_3D = np.pi / 3  # orientation constante du dernier segment

OMEGA_MAX = 2.0  # pour normaliser les vitesses angulaires


def generate_circular_trajectory(
    center: np.ndarray,
    radius: float,
    speed: float,
    n_steps: int,
    dt: float,
    phase: float = 0.0
) -> np.ndarray:
    """Génère une trajectoire circulaire dans l'espace de travail."""
    t = np.arange(n_steps) * dt + phase
    x = center[0] + radius * np.cos(speed * t)
    y = center[1] + radius * np.sin(speed * t)
    return np.stack([x, y], axis=1)


def arm_obs_from_angles_2dof(theta1: float, theta2: float, dtheta1: float, dtheta2: float) -> np.ndarray:
    """Construit l'observation bras 6D normalisée."""
    x = L1_2D * np.cos(theta1) + L2_2D * np.cos(theta1 + theta2)
    y = L1_2D * np.sin(theta1) + L2_2D * np.sin(theta1 + theta2)
    max_reach = L1_2D + L2_2D
    
    return np.array([
        theta1 / np.pi,
        theta2 / np.pi,
        dtheta1 / OMEGA_MAX,
        dtheta2 / OMEGA_MAX,
        x / max_reach,
        y / max_reach,
    ], dtype=np.float32)


def arm_obs_from_angles_3dof(theta1: float, theta2: float, theta3: float,
                              dtheta1: float, dtheta2: float, dtheta3: float) -> np.ndarray:
    """Construit l'observation bras 8D normalisée."""
    x = L1_3D * np.cos(theta1) + L2_3D * np.cos(theta1 + theta2) + L3_3D * np.cos(theta1 + theta2 + theta3)
    y = L1_3D * np.sin(theta1) + L2_3D * np.sin(theta1 + theta2) + L3_3D * np.sin(theta1 + theta2 + theta3)
    max_reach = L1_3D + L2_3D + L3_3D
    
    return np.array([
        theta1 / np.pi,
        theta2 / np.pi,
        theta3 / np.pi,
        dtheta1 / OMEGA_MAX,
        dtheta2 / OMEGA_MAX,
        dtheta3 / OMEGA_MAX,
        x / max_reach,
        y / max_reach,
    ], dtype=np.float32)


def collect_reaching_trajectories(
    n_pairs: int = N_PAIRS,
    seq_len: int = SEQ_LEN,
    dt: float = DT,
) -> dict:
    """
    Génère n_pairs de trajectoires de reaching synchronisées 2DoF ↔ 3DoF.
    Les deux bras poursuivent la même cible mobile avec des vitesses adaptées.
    """
    print("\n" + "="*60)
    print("Phase 1: Generating IK-based reaching trajectories")
    print("="*60)
    print(f"  Pairs: {n_pairs}")
    print(f"  Sequence length: {seq_len}")
    print(f"  DT: {dt}s ({1/dt} Hz)")

    rng = np.random.RandomState(42)
    
    all_arm2 = []
    all_arm3 = []
    
    for pair_idx in tqdm(range(n_pairs), desc="Generating trajectories"):
        # Paramètres aléatoires de la trajectoire
        center_radius = rng.uniform(0.2, 0.6)
        center_angle = rng.uniform(-np.pi, np.pi)
        center = center_radius * np.array([np.cos(center_angle), np.sin(center_angle)])
        
        radius = rng.uniform(0.1, 0.4)
        speed = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2*np.pi)
        
        # Trajectoire de la cible dans l'espace de travail (même pour les deux bras)
        target_traj = generate_circular_trajectory(
            center, radius, speed, seq_len, dt, phase
        )
        
        # 2DoF: IK direct
        theta1_2d = np.zeros(seq_len)
        theta2_2d = np.zeros(seq_len)
        prev_t1 = prev_t2 = 0.0
        
        for t in range(seq_len):
            t1, t2 = ik_2dof(target_traj[t], L1_2D, L2_2D)
            theta1_2d[t] = t1
            theta2_2d[t] = t2
        
        # Vitesses angulaires 2DoF
        dtheta1_2d = np.diff(theta1_2d, prepend=theta1_2d[0]) / dt
        dtheta2_2d = np.diff(theta2_2d, prepend=theta2_2d[0]) / dt
        
        # 3DoF: IK avec orientation fixée
        theta1_3d = np.zeros(seq_len)
        theta2_3d = np.zeros(seq_len)
        theta3_3d = np.zeros(seq_len)
        
        for t in range(seq_len):
            t1, t2, t3 = ik_3dof(target_traj[t], L1_3D, L2_3D, L3_3D, THETA_TOTAL_3D)
            theta1_3d[t] = t1
            theta2_3d[t] = t2
            theta3_3d[t] = t3
        
        dtheta1_3d = np.diff(theta1_3d, prepend=theta1_3d[0]) / dt
        dtheta2_3d = np.diff(theta2_3d, prepend=theta2_3d[0]) / dt
        dtheta3_3d = np.diff(theta3_3d, prepend=theta3_3d[0]) / dt
        
        # Construire les observations bras
        for t in range(seq_len):
            obs2 = arm_obs_from_angles_2dof(
                theta1_2d[t], theta2_2d[t],
                dtheta1_2d[t], dtheta2_2d[t]
            )
            obs3 = arm_obs_from_angles_3dof(
                theta1_3d[t], theta2_3d[t], theta3_3d[t],
                dtheta1_3d[t], dtheta2_3d[t], dtheta3_3d[t]
            )
            all_arm2.append(obs2)
            all_arm3.append(obs3)
    
    result = {
        'arm_states_2dof': np.stack(all_arm2, axis=0),
        'arm_states_3dof': np.stack(all_arm3, axis=0),
        'metadata': {
            'n_pairs': n_pairs,
            'seq_len': seq_len,
            'dt': dt,
            'total_samples': len(all_arm2),
            'source': 'ik_trajectories',
            'theta_total_3d': THETA_TOTAL_3D,
        }
    }
    
    print(f"\n  Generated {len(result['arm_states_2dof'])} paired samples")
    print(f"  arm_states_2dof shape: {result['arm_states_2dof'].shape}")
    print(f"  arm_states_3dof shape: {result['arm_states_3dof'].shape}")
    
    return result


def main():
    data_dir = Path("./data/LSUNN")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data = collect_reaching_trajectories()
    
    save_path = data_dir / "trajectories.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n  Trajectories saved → {save_path}")


if __name__ == "__main__":
    main()