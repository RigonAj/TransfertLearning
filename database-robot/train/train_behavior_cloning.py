from csv import writer
import os
import pickle
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.env_pushball_2dof import PushBallEnv_2dof

# clear tensorboard logs
log_dir = "./data/models/bc_pushball_2/tensorboard"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)


############################################################
# Dataset
############################################################

class DemonstrationDataset(Dataset):

    def __init__(self, demo_file):

        with open(demo_file, "rb") as f:
            episodes = pickle.load(f)

        obs = []
        actions = []

        for ep in episodes:
            obs.append(ep["obs"])
            actions.append(ep["actions"])

        self.obs = torch.tensor(
            np.concatenate(obs, axis=0),
            dtype=torch.float32
        )

        self.actions = torch.tensor(
            np.concatenate(actions, axis=0),
            dtype=torch.float32
        )

        print("Transitions :", len(self.obs))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


############################################################
# Training
############################################################

def train_bc(

        demonstrations_path,
        vecnormalize_path,

        output_dir="./database/models/bc_pushball",

        batch_size=512,
        epochs=350,
        learning_rate=3e-4,
        checkpoint_every=50,

):

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "tensorboard")

    writer = SummaryWriter(log_dir)

    ########################################################

    dataset = DemonstrationDataset(demonstrations_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    ########################################################
    # Dummy PPO (uniquement pour récupérer le réseau)
    ########################################################

    env = DummyVecEnv([lambda: PushBallEnv_2dof(None)])

    env = VecNormalize.load(vecnormalize_path, env)

    env.training = False
    env.norm_reward = False

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        verbose=0,
        device="auto"
    )

    optimizer = torch.optim.Adam(
        model.policy.parameters(),
        lr=learning_rate
    )

    ########################################################
    # BC
    ########################################################

    best_loss = np.inf

    global_step = 0

    pbar = tqdm(range(epochs), desc="Training BC")

    for epoch in pbar:

        epoch_loss = 0.0

        for obs, actions in loader:

            obs = obs.to(model.device)
            actions = actions.to(model.device)

            dist = model.policy.get_distribution(obs)

            pred_actions = dist.distribution.mean

            loss = F.mse_loss(pred_actions, actions)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # tensorboard
            writer.add_scalar("loss/train", loss.item(), global_step)

        epoch_loss /= len(loader)

        writer.add_scalar("loss/epoch", epoch_loss, epoch)

        #print(f"Epoch {epoch+1:3d} | loss = {epoch_loss:.6f}")
        pbar.set_postfix({"loss": f"{epoch_loss:.6f}"})

        ####################################################
        # Save best model
        ####################################################

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            model.save(
                os.path.join(output_dir, "best_model")
            )

            shutil.copy(
                vecnormalize_path,
                os.path.join(output_dir, "vec_normalize.pkl")
            )

            #print("Best model saved.")
        
        if (epoch + 1) % checkpoint_every == 0:

            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")

            model.save(checkpoint_path)

            #print(f"Checkpoint saved at epoch {epoch+1}")

    writer.close()
    
    print("\nTraining finished.")
    print("Best loss :", best_loss)
    print(f"Best model saved at: {os.path.join(output_dir, 'best_model.zip')}")


############################################################

if __name__ == "__main__":

    train_bc(

        demonstrations_path="./database/pushball_2dof/demonstrations.pkl",

        vecnormalize_path="./database/models/ppo_pushball_2dof_1/vec_normalize.pkl",

        output_dir="./data/models/bc_pushball_2",

        epochs=500,
        checkpoint_every=50,
        batch_size=512,
        learning_rate=3e-4

    )