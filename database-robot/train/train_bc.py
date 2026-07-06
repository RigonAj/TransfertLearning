"""Behavior cloning unifié — remplace train_behavior_cloning.py et train_bc_imitation.py.

Verdict sur les deux anciennes versions :
  - train_behavior_cloning.py : la meilleure base. Boucle propre (DataLoader),
    MSE sur la moyenne de la distribution de la politique SB3. Défauts : pas de
    validation (le "best model" est choisi sur la loss d'ENTRAÎNEMENT, donc
    biaisé vers l'overfitting), pas de scheduler, pas de seed.
  - train_bc_imitation.py : importe la lib `imitation` (BC, Transitions) sans
    jamais l'utiliser — elle construit des Transitions puis refait la même
    boucle MSE à la main, avec un batching manuel bugué (dernier batch vide si
    dataset_size % batch_size == 0) et un `n_batches = size // batch + 1` qui
    biaise la moyenne de la loss. Aucune raison de la conserver.

Ce script ajoute :
  - split train/validation PAR ÉPISODE (pas par transition : deux transitions
    du même épisode sont corrélées, les mettre de part et d'autre du split
    rend la validation optimiste) ; best model choisi sur la loss de validation ;
  - loss au choix : --loss mse (défaut, MSE sur la moyenne déterministe) ou
    --loss nll (log-vraisemblance gaussienne : apprend aussi log_std, utile si
    la politique clonée est ensuite utilisée en mode stochastique) ;
  - scheduler ReduceLROnPlateau + early stopping sur la validation ;
  - seed, gradient clipping, contrôle de cohérence de normalisation des obs ;
  - sauvegarde inchangée : format SB3 (`ppo.save`) + copie de vec_normalize.pkl,
    compatible avec test/visualize/replay_bc_pushball.py.

Usage :
    cd database-robot && python3 train/train_bc.py \
        --demos database/pushball_2dof/demonstrations.pkl \
        --vecnorm database/models/ppo_pushball_2dof_1/vec_normalize.pkl \
        --out data/models/bc_pushball_unified --env pushball_2dof
"""

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

# Rend `envs` importable quel que soit le répertoire de lancement
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env(name):
    if name == "pushball_2dof":
        from envs.env_pushball_2dof import PushBallEnv_2dof
        return lambda: PushBallEnv_2dof(None)
    if name == "pushball_3dof":
        from envs.env_pushball_3dof import PushBallEnv_3dof
        return lambda: PushBallEnv_3dof(None)
    raise ValueError(f"env inconnu: {name}")


def load_episodes(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def split_episodes(episodes, val_frac, seed):
    """Split par épisode : toutes les transitions d'un épisode restent ensemble."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(round(val_frac * len(episodes))))
    val_ids = set(order[:n_val].tolist())
    train_eps = [e for i, e in enumerate(episodes) if i not in val_ids]
    val_eps = [e for i, e in enumerate(episodes) if i in val_ids]
    return train_eps, val_eps


def to_tensors(episodes):
    obs = torch.tensor(np.concatenate([e["obs"] for e in episodes]), dtype=torch.float32)
    acts = torch.tensor(np.concatenate([e["actions"] for e in episodes]), dtype=torch.float32)
    return obs, acts


def check_obs_normalization(obs, vecnorm):
    """Heuristique : les obs des démos sont-elles déjà normalisées VecNormalize ?

    Les obs brutes des envs sont physiquement normalisées dans [-1, 1] ; après
    VecNormalize elles ont un écart-type ~1 par dimension mais dépassent ±1.
    """
    std = obs.std(dim=0).mean().item()
    amax = obs.abs().max().item()
    looks_normalized = amax > 1.5 and 0.5 < std < 2.0
    print(f"Obs démos : std moyen={std:.3f}, |max|={amax:.3f} -> "
          f"{'déjà normalisées VecNormalize (OK, ne pas re-normaliser)' if looks_normalized else 'brutes (utiliser --normalize-obs)'}")
    return looks_normalized


def batch_loss(policy, obs, acts, loss_kind, ent_weight, l2_weight):
    dist = policy.get_distribution(obs)
    if loss_kind == "mse":
        loss = F.mse_loss(dist.distribution.mean, acts)
    else:  # nll
        loss = -dist.log_prob(acts).mean()
        if ent_weight > 0:
            loss = loss - ent_weight * dist.entropy().mean()
    if l2_weight > 0:
        l2 = sum(p.pow(2).sum() for p in policy.parameters())
        loss = loss + l2_weight * l2
    return loss


def main():
    p = argparse.ArgumentParser(description="Behavior cloning unifié (SB3 backbone)")
    p.add_argument("--demos", default="./database/pushball_2dof/demonstrations.pkl")
    p.add_argument("--vecnorm", default="./database/models/ppo_pushball_2dof_1/vec_normalize.pkl")
    p.add_argument("--out", default="./data/models/bc_pushball_unified")
    p.add_argument("--env", default="pushball_2dof",
                   choices=["pushball_2dof", "pushball_3dof"])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--loss", choices=["mse", "nll"], default="mse")
    p.add_argument("--ent-weight", type=float, default=1e-3,
                   help="poids d'entropie (loss nll uniquement)")
    p.add_argument("--l2-weight", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=25, help="early stopping (époques)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normalize-obs", action="store_true",
                   help="appliquer VecNormalize aux obs des démos (si enregistrées brutes)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    log_dir = os.path.join(args.out, "tensorboard")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    # ── Env + VecNormalize (stats du run PPO d'origine) ───────────────
    env = DummyVecEnv([make_env(args.env)])
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False
    env.norm_reward = False

    # ── Données ───────────────────────────────────────────────────────
    episodes = load_episodes(args.demos)
    train_eps, val_eps = split_episodes(episodes, args.val_frac, args.seed)
    train_obs, train_acts = to_tensors(train_eps)
    val_obs, val_acts = to_tensors(val_eps)
    print(f"Épisodes: {len(train_eps)} train / {len(val_eps)} val | "
          f"Transitions: {len(train_obs)} train / {len(val_obs)} val")

    already_normalized = check_obs_normalization(train_obs, env)
    if args.normalize_obs:
        if already_normalized:
            print("ATTENTION: --normalize-obs demandé mais les obs semblent déjà "
                  "normalisées — double normalisation probable.")
        train_obs = torch.tensor(env.normalize_obs(train_obs.numpy()), dtype=torch.float32)
        val_obs = torch.tensor(env.normalize_obs(val_obs.numpy()), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_obs, train_acts),
                              batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(val_obs, val_acts),
                            batch_size=args.batch_size, shuffle=False)

    # ── Politique (backbone SB3, pour compat replay/PPO.load) ─────────
    ppo = PPO("MlpPolicy", env, verbose=0, device="auto", seed=args.seed)
    policy = ppo.policy
    device = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    # ── Boucle ────────────────────────────────────────────────────────
    best_val = float("inf")
    bad_epochs = 0
    global_step = 0
    best_path = os.path.join(args.out, "best_model")

    pbar = tqdm(range(args.epochs), desc=f"Training BC ({args.loss})")
    for epoch in pbar:
        policy.set_training_mode(True)
        train_loss = 0.0
        for obs, acts in train_loader:
            obs, acts = obs.to(device), acts.to(device)
            loss = batch_loss(policy, obs, acts, args.loss, args.ent_weight, args.l2_weight)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            global_step += 1
            writer.add_scalar("loss/train", loss.item(), global_step)
        train_loss /= len(train_loader)

        policy.set_training_mode(False)
        val_loss = 0.0
        with torch.no_grad():
            for obs, acts in val_loader:
                obs, acts = obs.to(device), acts.to(device)
                val_loss += batch_loss(policy, obs, acts, args.loss,
                                       args.ent_weight, args.l2_weight).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        writer.add_scalar("loss/epoch_train", train_loss, epoch)
        writer.add_scalar("loss/epoch_val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        pbar.set_postfix({"train": f"{train_loss:.6f}", "val": f"{val_loss:.6f}",
                          "best": f"{best_val:.6f}"})

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            ppo.save(best_path)
            shutil.copy(args.vecnorm, os.path.join(args.out, "vec_normalize.pkl"))
        else:
            bad_epochs += 1
            if bad_epochs > args.patience:
                print(f"\nEarly stopping à l'époque {epoch + 1} "
                      f"(pas d'amélioration val depuis {args.patience} époques).")
                break

        if (epoch + 1) % args.checkpoint_every == 0:
            ppo.save(os.path.join(args.out, f"checkpoint_epoch_{epoch + 1}"))

    writer.close()
    print("\nTraining finished.")
    print(f"Best val loss : {best_val:.6f}")
    print(f"Best model    : {best_path}.zip (+ vec_normalize.pkl)")


if __name__ == "__main__":
    main()
