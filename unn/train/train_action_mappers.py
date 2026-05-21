"""
Train UNN action mappers: (joint_angles, desired_ee_disp) → joint_action
"""

import sys
import pickle
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unn.bases_unn import ActionMapper, train_action_mapper

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("./data/UNN/trajectories_unn.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 2DoF mapper
    print("\n--- Training 2DoF action mapper ---")
    mapper2 = ActionMapper(n_joints=2)
    mapper2 = train_action_mapper(
        mapper2,
        joint_angles=data['2dof']['joint_norm'],
        desired_disp=data['2dof']['ee_disp_norm'],
        target_actions=data['2dof']['action'],
        epochs=200, batch_size=256, lr=1e-3, device=device
    )
    save_dir = Path("./data/UNN/action_mappers")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mapper2.state_dict(), save_dir / "mapper_2dof.pt")

    # 3DoF mapper
    print("\n--- Training 3DoF action mapper ---")
    mapper3 = ActionMapper(n_joints=3)
    mapper3 = train_action_mapper(
        mapper3,
        joint_angles=data['3dof']['joint_norm'],
        desired_disp=data['3dof']['ee_disp_norm'],
        target_actions=data['3dof']['action'],
        epochs=200, batch_size=256, lr=1e-3, device=device
    )
    torch.save(mapper3.state_dict(), save_dir / "mapper_3dof.pt")
    print("Done.")

if __name__ == "__main__":
    main()
