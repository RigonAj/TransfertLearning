import os
import pickle
import shutil
import numpy as np
from tqdm import tqdm

import torch as th
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from imitation.algorithms.bc import BC
from imitation.data.types import Transitions

from envs.env_pushball_2dof import PushBallEnv_2dof

# clear tensorboard logs
log_dir = "./data/models/bc_pushball_1/tensorboard"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)


############################################################
# DATASET
############################################################

def load_transitions(demonstrations_path):

    with open(demonstrations_path, "rb") as f:
        episodes = pickle.load(f)

    obs, acts, next_obs, dones = [], [], [], []

    for ep in episodes:
        o = ep["obs"]
        a = ep["actions"]
        d = ep["dones"]

        obs.append(o)
        acts.append(a)

        n = np.vstack([o[1:], o[-1:]])
        next_obs.append(n)
        dones.append(d)

    obs = np.concatenate(obs).astype(np.float32)
    acts = np.concatenate(acts).astype(np.float32)
    next_obs = np.concatenate(next_obs).astype(np.float32)
    # ValueError: dones must be 1D array, one entry for each timestep: (106538, 1) != (106538,) 
    dones = np.concatenate(dones).all(axis=1).astype(np.bool_)

    return Transitions(
        obs=obs,
        acts=acts,
        next_obs=next_obs,
        dones=dones,
        infos=np.array([{} for _ in range(len(obs))], dtype=object),
    )


############################################################
# TRAINING LOOP
############################################################

def train_bc(
    demonstrations_path,
    vecnormalize_path,
    output_dir="./database/models/bc_pushball",
    batch_size=512,
    epochs=350,
    checkpoint_every=50,
    lr=3e-4,
):

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "tensorboard")

    writer = SummaryWriter(log_dir)

    ########################################################
    # DATA
    ########################################################

    transitions = load_transitions(demonstrations_path)

    obs = torch_obs = th.tensor(transitions.obs, dtype=th.float32)
    acts = th.tensor(transitions.acts, dtype=th.float32)

    dataset_size = len(obs)

    print(f"Transitions: {dataset_size}")

    ########################################################
    # ENV + POLICY (SB3 backbone)
    ########################################################

    env = DummyVecEnv([lambda: PushBallEnv_2dof(None)])
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False
    env.norm_reward = False

    ppo = PPO("MlpPolicy", env, verbose=0, device="auto")

    policy = ppo.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=lr)

    device = policy.device

    ########################################################
    # TRAIN LOOP
    ########################################################

    best_loss = 1e9

    n_batches_per_epoch = dataset_size // batch_size + 1
    global_step = 0

    pbar = tqdm(range(epochs), desc="Training BC")

    for epoch in pbar:

        perm = th.randperm(dataset_size)

        epoch_loss = 0.0


        for b in range(n_batches_per_epoch):

            idx = perm[b * batch_size : (b + 1) * batch_size]

            batch_obs = obs[idx].to(device)
            batch_act = acts[idx].to(device)

            dist = policy.get_distribution(batch_obs)
            pred = dist.distribution.mean

            loss = F.mse_loss(pred, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            ####################################################
            # TensorBoard 
            ####################################################

            writer.add_scalar("loss/train", loss.item(), global_step)

        epoch_loss /= n_batches_per_epoch

        writer.add_scalar("loss/epoch", epoch_loss, epoch)

        #print(f"Epoch {epoch+1} | loss = {epoch_loss:.6f}")
        pbar.set_postfix({"loss": f"{epoch_loss:.6f}"})

        ########################################################
        # SAVE BEST MODEL
        ########################################################

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            save_path = os.path.join(output_dir, "best_model")

            ppo.save(save_path)

            shutil.copy(
                vecnormalize_path,
                os.path.join(output_dir, "vec_normalize.pkl")
            )

            #print(f"Best model saved (loss={best_loss:.6f})")
        
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
            ppo.save(checkpoint_path)
            #print(f"Checkpoint saved at epoch {epoch+1}")

    writer.close()

    print("\nTraining finished")
    print("Best loss:", best_loss)
    print(f"Best model saved at: {save_path}")


############################################################

if __name__ == "__main__":

    train_bc(
        demonstrations_path="./database/pushball_2dof/demonstrations.pkl",
        vecnormalize_path="./database/models/ppo_pushball_2dof_1/vec_normalize.pkl",
        output_dir="./data/models/bc_pushball_1",
        batch_size=512,
        epochs=500,
        checkpoint_every=50,
        lr=3e-4,
    )