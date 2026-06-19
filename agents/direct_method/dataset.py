
import numpy as np
from torch.utils.data import Dataset



class TrajectoriesTrainingDataset(Dataset):
    def __init__(self, trajectories_r1, trajectories_r2, dim_R1, dim_R2):
        # Load trajectories from .txt files
        # --Load trajectories robot 1
        self.trajectories_r1 = np.loadtxt(trajectories_r1,dtype=np.float32)
        # revome the last two columns(x,y) end effector position
        self.trajectories_r1 = self.trajectories_r1[:, :dim_R1]
        
        # --Load trajectories robot 2
        self.trajectories_r2 = np.loadtxt(trajectories_r2,dtype=np.float32)
        # revome the last two columns(x,y) end effector position
        self.trajectories_r2 = self.trajectories_r2[:, :dim_R2]

        vel_r1 = self.trajectories_r1[:, dim_R1 // 2:]
        vel_r2 = self.trajectories_r2[:, dim_R2 // 2:]
        # Create validity mask
        valid_mask_r1 = np.all((vel_r1 >= -2.0) & (vel_r1 <= 2.0), axis=1)
        valid_mask_r2 = np.all((vel_r2 >= -2.0) & (vel_r2 <= 2.0), axis=1)
        valid_mask = valid_mask_r1 & valid_mask_r2

        self.trajectories_r1 = self.trajectories_r1[valid_mask]
        self.trajectories_r2 = self.trajectories_r2[valid_mask]

        # normalize joints positions and velocities
        self.trajectories_r1[:, :dim_R1 // 2] = self.trajectories_r1[:, :dim_R1 // 2]/3.14
        self.trajectories_r2[:, :dim_R2 // 2] = self.trajectories_r2[:, :dim_R2 // 2]/3.14
        self.trajectories_r1[:, dim_R1 // 2:] = self.trajectories_r1[:, dim_R1 // 2:]/2.0
        self.trajectories_r2[:, dim_R2 // 2:] = self.trajectories_r2[:, dim_R2 // 2:]/2.0
        
    def __len__(self):
        return len(self.trajectories_r1)

    def __getitem__(self, idx):
        sample = (self.trajectories_r1[idx], self.trajectories_r2[idx])
        return sample
    

class TrajectoriesTrainingActionDataset(Dataset):
    def __init__(self, trajectories_r1, trajectories_r2, action_dim_R1, action_dim_R2):
        # Load trajectories from .txt files
        # --Load trajectories robot 1
        self.trajectories_r1 = np.loadtxt(trajectories_r1,dtype=np.float32)
        # revome the last two columns(x,y) end effector position
        self.trajectories_r1 = self.trajectories_r1[:, :action_dim_R1]
        
        # --Load trajectories robot 2
        self.trajectories_r2 = np.loadtxt(trajectories_r2,dtype=np.float32)
        # revome the last two columns(x,y) end effector position
        self.trajectories_r2 = self.trajectories_r2[:, :action_dim_R2]

        # normalize joints positions and velocities
        self.trajectories_r1 = self.trajectories_r1/2.0
        self.trajectories_r2 = self.trajectories_r2/2.0
        
    def __len__(self):
        return len(self.trajectories_r1)

    def __getitem__(self, idx):
        sample = (self.trajectories_r1[idx], self.trajectories_r2[idx])
        return sample