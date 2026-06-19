# lsunn/train/train_joint_transfer.py
"""
Entraînement simultané avec optimisation mémoire
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import pickle
import gc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof
from lsunn.bases_vae import BaseVAE, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM

# Dimensions
ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8
ACTION_DIM_2DOF = 2
ACTION_DIM_3DOF = 3


class LatentEnv(gym.Wrapper):
    """Wrapper qui expose l'espace latent."""
    def __init__(self, env, base_vae, device="cpu", latent_dim=16):
        super().__init__(env)
        self.base_vae = base_vae
        self.device = device
        self.base_vae.eval()
        self.observation_space = spaces.Box(-10, 10, (latent_dim,), dtype=np.float32)
    
    @torch.no_grad()
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        latent = self.base_vae.encode(obs_t).detach().cpu().numpy().flatten()
        return latent, reward, terminated, truncated, info
    
    @torch.no_grad()
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        latent = self.base_vae.encode(obs_t).detach().cpu().numpy().flatten()
        return latent, info


class ActionMapper(nn.Module):
    """Mapper d'action pour transfert bidirectionnel."""
    def __init__(self, state_dim, act_in_dim, act_out_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.act_in_dim = act_in_dim
        self.act_out_dim = act_out_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_out_dim),
            nn.Tanh(),
        )
    
    def forward(self, state, action):
        # Vérifier les dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Vérifier que les dimensions correspondent
        assert state.shape[1] == self.state_dim, f"Expected state dim {self.state_dim}, got {state.shape[1]}"
        assert action.shape[1] == self.act_in_dim, f"Expected action dim {self.act_in_dim}, got {action.shape[1]}"
        
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class JointTransferTrainer:
    """
    Entraînement complet LS-UNN sans politiques pré-entraînées.
    """
    
    def __init__(self, latent_dim=16, device="cpu"):
        self.device = device
        self.latent_dim = latent_dim
        
        # Réduire la taille des couches cachées pour économiser la mémoire
        hidden_dim = 128
        self.base_2dof = BaseVAE(10, latent_dim, hidden_dim).to(device)
        self.base_3dof = BaseVAE(12, latent_dim, hidden_dim).to(device)
        
        self.ppo_2dof = None
        self.ppo_3dof = None
        self.mapper_2to3 = None
        self.mapper_3to2 = None
        self.vec_norm_2dof = None
        self.vec_norm_3dof = None
    
    # ================================================================
    # Phase 1: Trajectoires aléatoires
    # ================================================================
    
    def collect_random_paired_trajectories(self, n_pairs=10000, seq_len=30):
        """Collecte de paires avec actions aléatoires."""
        print("\n" + "="*60)
        print("Phase 1: Collecting random paired trajectories")
        print("="*60)
        
        env2 = PushBallEnv_2dof(render_mode=None, max_steps=1000000)
        env3 = PushBallEnv_3dof(render_mode=None, max_steps=1000000)
        rng = np.random.RandomState(42)
        max_reach = env2.max_reach
        
        states2_full, states3_full, arm_states2, arm_states3, acts2, acts3 = [], [], [], [], [], []
        
        for _ in tqdm(range(n_pairs), desc="Random pairs"):
            # Même config initiale
            target = rng.uniform(0.4, 0.75, 2) * max_reach
            angle = rng.uniform(-np.pi, np.pi)
            target = np.array([target[0]*np.cos(angle), target[0]*np.sin(angle)])
            
            # Balle à distance > 0.3
            for _ in range(100):
                ball = rng.uniform(0.3, 0.75, 2) * max_reach
                angle_b = rng.uniform(-np.pi, np.pi)
                ball = np.array([ball[0]*np.cos(angle_b), ball[0]*np.sin(angle_b)])
                if np.linalg.norm(ball - target) >= 0.3:
                    break
            
            env2.reset()
            env3.reset()
            env2.target, env2.ball = target.copy(), ball.copy()
            env3.target, env3.ball = target.copy(), ball.copy()
            
            traj2_full, traj3_full, traj2_arm, traj3_arm, act_traj2, act_traj3 = [], [], [], [], [], []
            
            for _ in range(seq_len):
                a2 = rng.uniform(-1, 1, 2).astype(np.float32)
                a3 = rng.uniform(-1, 1, 3).astype(np.float32)
                
                obs2, _, _, _, _ = env2.step(a2)
                obs3, _, _, _, _ = env3.step(a3)
                
                traj2_full.append(obs2)
                traj3_full.append(obs3)
                traj2_arm.append(obs2[:ARM_OBS_2DOF])
                traj3_arm.append(obs3[:ARM_OBS_3DOF])
                act_traj2.append(a2)
                act_traj3.append(a3)
            
            states2_full.append(np.stack(traj2_full))
            states3_full.append(np.stack(traj3_full))
            arm_states2.append(np.stack(traj2_arm))
            arm_states3.append(np.stack(traj3_arm))
            acts2.append(np.stack(act_traj2))
            acts3.append(np.stack(act_traj3))
        
        env2.close()
        env3.close()
        
        # full observation dims (including task info)
        s2_full_dim = env2.observation_space.shape[0]
        s3_full_dim = env3.observation_space.shape[0]

        result = {
            'states_2dof': np.stack(states2_full).reshape(-1, s2_full_dim),
            'states_3dof': np.stack(states3_full).reshape(-1, s3_full_dim),
            'arm_states_2dof': np.stack(arm_states2).reshape(-1, ARM_OBS_2DOF),
            'arm_states_3dof': np.stack(arm_states3).reshape(-1, ARM_OBS_3DOF),
            'actions_2dof': np.stack(acts2).reshape(-1, ACTION_DIM_2DOF),
            'actions_3dof': np.stack(acts3).reshape(-1, ACTION_DIM_3DOF),
        }
        
        # Nettoyer la mémoire
        del states2_full, states3_full, arm_states2, arm_states3, acts2, acts3
        gc.collect()
        
        return result
    
    # ================================================================
    # Phase 2: Entraînement VAE
    # ================================================================
    
    def train_vae(self, data, epochs=150):
        """Entraîne les VAE pour aligner les espaces latents."""
        print("\n" + "="*60)
        print("Phase 2: Training VAE bases (shared latent space)")
        print("="*60)
        
        # Utiliser moins de données pour l'entraînement
        n_samples = min(50000, len(data['states_2dof']))
        indices = np.random.choice(len(data['states_2dof']), n_samples, replace=False)
        
        s2 = torch.tensor(data['states_2dof'][indices], dtype=torch.float32, device=self.device)
        s3 = torch.tensor(data['states_3dof'][indices], dtype=torch.float32, device=self.device)
        
        optimizer = optim.Adam(
            list(self.base_2dof.parameters()) + list(self.base_3dof.parameters()),
            lr=3e-4
        )
        recon_crit = nn.MSELoss()
        sim_crit = nn.MSELoss()
        
        pbar = tqdm(range(epochs), desc="VAE Training")
        for epoch in pbar:
            perm = torch.randperm(len(s2))
            epoch_loss = 0.0
            
            for i in range(0, len(s2), 256):
                idx = perm[i:i+256]
                b2 = s2[idx]
                b3 = s3[idx]
                
                z2, mu2, logvar2 = self.base_2dof.encoder(b2)
                z3, mu3, logvar3 = self.base_3dof.encoder(b3)
                
                recon2 = self.base_2dof.decoder(z2)
                recon3 = self.base_3dof.decoder(z3)
                cross2 = self.base_2dof.decoder(z3)
                cross3 = self.base_3dof.decoder(z2)
                
                loss = (10.0 * (recon_crit(recon2, b2) + recon_crit(recon3, b3)) +
                        0.001 * (-0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp()) / b2.shape[0] +
                                 -0.5 * torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp()) / b3.shape[0]) +
                        1.0 * sim_crit(z2, z3) +
                        1.0 * (recon_crit(cross2, b2) + recon_crit(cross3, b3)))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            denom = max(1, (len(s2) + 256 - 1) // 256)
            pbar.set_postfix({'loss': f'{epoch_loss/denom:.4f}'})
        
        print("  VAE training complete")
        
        # Nettoyer la mémoire
        del s2, s3
        gc.collect()
    
    # ================================================================
    # Phase 3: Entraînement PPO dans espace latent
    # ================================================================
    
    def train_latent_policies(self, total_timesteps=5_000_000):
        """Entraîne les politiques PPO pour 2DoF et 3DoF dans l'espace latent."""
        print("\n" + "="*60)
        print("Phase 3: Training PPO policies in shared latent space")
        print("="*60)
        
        # Réduire le nombre d'environnements parallèles
        n_envs = 8
        
        # Pour 2DoF
        print("\n  Training 2DoF latent policy...")
        self.ppo_2dof, self.vec_norm_2dof = self._train_single_policy(
            self.base_2dof, PushBallEnv_2dof, total_timesteps // 2, "lsunn_2dof_latent", n_envs
        )
        
        # Pour 3DoF
        print("\n  Training 3DoF latent policy...")
        self.ppo_3dof, self.vec_norm_3dof = self._train_single_policy(
            self.base_3dof, PushBallEnv_3dof, total_timesteps // 2, "lsunn_3dof_latent", n_envs
        )
    
    def _train_single_policy(self, base_vae, EnvClass, timesteps, run_id, n_envs=8):
        """Entraîne une politique PPO dans l'espace latent."""
        
        def make_env():
            raw_env = EnvClass(render_mode=None, max_steps=150)
            return LatentEnv(raw_env, base_vae, self.device, self.latent_dim)
        
        train_env = SubprocVecEnv([lambda: Monitor(make_env()) for _ in range(n_envs)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        model_dir = Path(f"./data/LSUNN/{run_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        ppo = PPO("MlpPolicy", train_env,
                  n_steps=1024,
                  batch_size=256,
                  n_epochs=5,
                  learning_rate=3e-4, gamma=0.99,
                  policy_kwargs=dict(net_arch=[128, 128]),
                  device=self.device)
        
        ppo.learn(total_timesteps=timesteps, progress_bar=True)
        
        ppo.save(model_dir / "policy")
        train_env.save(model_dir / "vec_normalize.pkl")
        
        return ppo, train_env
    
    # ================================================================
    # Phase 4: Entraînement des mappers
    # ================================================================
    
    def train_mappers(self, data, epochs=200):
        """Entraîne les mappers pour transfert bidirectionnel."""
        print("\n" + "="*60)
        print("Phase 4: Training action mappers for bidirectional transfer")
        print("="*60)
        
        # Utiliser moins de données
        n_samples = min(50000, len(data['arm_states_2dof']))
        indices = np.random.choice(len(data['arm_states_2dof']), n_samples, replace=False)
        
        # Use arm-only observations for mapper training
        s2 = torch.tensor(data['arm_states_2dof'][indices], dtype=torch.float32, device=self.device)
        s3 = torch.tensor(data['arm_states_3dof'][indices], dtype=torch.float32, device=self.device)
        a2 = torch.tensor(data['actions_2dof'][indices], dtype=torch.float32, device=self.device)
        a3 = torch.tensor(data['actions_3dof'][indices], dtype=torch.float32, device=self.device)
        
        # Mapper 2DoF → 3DoF (prend état 3DoF et action 2DoF, sort action 3DoF)
        print("\n  Training mapper 2→3...")
        print(f"    Input: state_dim={ARM_OBS_3DOF}, act_in_dim={ACTION_DIM_2DOF}, act_out_dim={ACTION_DIM_3DOF}")
        self.mapper_2to3 = ActionMapper(ARM_OBS_3DOF, ACTION_DIM_2DOF, ACTION_DIM_3DOF, hidden_dim=256).to(self.device)
        self._train_single_mapper(self.mapper_2to3, s3, a2, a3, epochs, "mapper_2to3")
        
        # Mapper 3DoF → 2DoF (prend état 2DoF et action 3DoF, sort action 2DoF)
        print("\n  Training mapper 3→2...")
        print(f"    Input: state_dim={ARM_OBS_2DOF}, act_in_dim={ACTION_DIM_3DOF}, act_out_dim={ACTION_DIM_2DOF}")
        self.mapper_3to2 = ActionMapper(ARM_OBS_2DOF, ACTION_DIM_3DOF, ACTION_DIM_2DOF, hidden_dim=256).to(self.device)
        self._train_single_mapper(self.mapper_3to2, s2, a3, a2, epochs, "mapper_3to2")
    
    def _train_single_mapper(self, mapper, states, actions_in, actions_out, epochs, name):
        """Entraîne un mapper individuel."""
        optimizer = optim.Adam(mapper.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        n_samples = len(states)
        split = int(n_samples * 0.9)
        
        train_idx = torch.randperm(n_samples)
        train_s = states[train_idx[:split]]
        train_a_in = actions_in[train_idx[:split]]
        train_a_out = actions_out[train_idx[:split]]
        val_s = states[train_idx[split:]]
        val_a_in = actions_in[train_idx[split:]]
        val_a_out = actions_out[train_idx[split:]]
        
        best_val_loss = float('inf')
        pbar = tqdm(range(epochs), desc=name)
        
        for epoch in pbar:
            # Forward
            pred = mapper(train_s, train_a_in)
            loss = criterion(pred, train_a_out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            mapper.eval()
            with torch.no_grad():
                val_pred = mapper(val_s, val_a_in)
                val_loss = criterion(val_pred, val_a_out)
            mapper.train()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            pbar.set_postfix({'train': f'{loss.item():.4f}', 'val': f'{val_loss.item():.4f}'})
        
        print(f"    Best val loss: {best_val_loss:.6f}")
    
    # ================================================================
    # Sauvegarde et chargement
    # ================================================================
    
    def save(self, save_dir="./data/LSUNN/joint_model"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.base_2dof.state_dict(), save_path / "base_2dof.pt")
        torch.save(self.base_3dof.state_dict(), save_path / "base_3dof.pt")
        
        if self.mapper_2to3:
            torch.save(self.mapper_2to3.state_dict(), save_path / "mapper_2to3.pt")
        if self.mapper_3to2:
            torch.save(self.mapper_3to2.state_dict(), save_path / "mapper_3to2.pt")
        
        # Sauvegarde des politiques PPO et VecNormalize
        if self.ppo_2dof:
            self.ppo_2dof.save(save_path / "latent_policy_2dof")
        if self.ppo_3dof:
            self.ppo_3dof.save(save_path / "latent_policy_3dof")
        if self.vec_norm_2dof:
            self.vec_norm_2dof.save(save_path / "vec_normalize_2dof.pkl")
        if self.vec_norm_3dof:
            self.vec_norm_3dof.save(save_path / "vec_normalize_3dof.pkl")
        
        print(f"  Model saved to {save_dir}")
    
    def load(self, save_dir="./data/LSUNN/joint_model"):
        save_path = Path(save_dir)
        
        self.base_2dof.load_state_dict(torch.load(save_path / "base_2dof.pt", map_location=self.device))
        self.base_3dof.load_state_dict(torch.load(save_path / "base_3dof.pt", map_location=self.device))
        
        mapper_2to3_path = save_path / "mapper_2to3.pt"
        if mapper_2to3_path.exists():
            self.mapper_2to3 = ActionMapper(ARM_OBS_3DOF, ACTION_DIM_2DOF, ACTION_DIM_3DOF, hidden_dim=256).to(self.device)
            self.mapper_2to3.load_state_dict(torch.load(mapper_2to3_path, map_location=self.device))
        
        mapper_3to2_path = save_path / "mapper_3to2.pt"
        if mapper_3to2_path.exists():
            self.mapper_3to2 = ActionMapper(ARM_OBS_2DOF, ACTION_DIM_3DOF, ACTION_DIM_2DOF, hidden_dim=256).to(self.device)
            self.mapper_3to2.load_state_dict(torch.load(mapper_3to2_path, map_location=self.device))
        
        print(f"  Model loaded from {save_dir}")
        return self
    
    # ================================================================
    # Test de transfert
    # ================================================================
    
    @torch.no_grad()
    def test_transfer_2to3(self, n_episodes=100):
        """Teste le transfert de 2DoF vers 3DoF."""
        return self._test_transfer_direction(
            source_policy=self.ppo_2dof,
            source_vec_norm=self.vec_norm_2dof,
            target_base=self.base_3dof,
            target_env_class=PushBallEnv_3dof,
            mapper=self.mapper_2to3,
            n_episodes=n_episodes,
            direction="2→3"
        )
    
    @torch.no_grad()
    def test_transfer_3to2(self, n_episodes=100):
        """Teste le transfert de 3DoF vers 2DoF."""
        return self._test_transfer_direction(
            source_policy=self.ppo_3dof,
            source_vec_norm=self.vec_norm_3dof,
            target_base=self.base_2dof,
            target_env_class=PushBallEnv_2dof,
            mapper=self.mapper_3to2,
            n_episodes=n_episodes,
            direction="3→2"
        )
    
    @torch.no_grad()
    def _test_transfer_direction(self, source_policy, source_vec_norm, target_base,
                                 target_env_class, mapper, n_episodes, direction):
        """Teste une direction de transfert."""
        print(f"\n  Testing transfer {direction}...")
        
        env = target_env_class(render_mode=None, max_steps=150)
        successes = 0
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                # 1. Encodage avec VAE cible
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                z = target_base.encode(obs_t).detach().cpu().numpy().flatten()
                
                # 2. Normalisation avec stats source
                z_norm = source_vec_norm.normalize_obs(z)
                
                # 3. Politique source → action source
                a_src, _ = source_policy.predict(z_norm, deterministic=True)
                
                # 4. Mapper → action cible
                # Extraire l'état du bras (les premières dimensions)
                if direction == "2→3":
                    # Pour transfert 2→3, l'état cible est l'état 3DoF (8 dimensions)
                    arm_obs = obs[:ARM_OBS_3DOF]
                else:
                    # Pour transfert 3→2, l'état cible est l'état 2DoF (6 dimensions)
                    arm_obs = obs[:ARM_OBS_2DOF]
                
                s_t = torch.tensor(arm_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                a_t = torch.tensor(a_src, dtype=torch.float32, device=self.device)
                if a_t.ndim == 1:
                    a_t = a_t.unsqueeze(0)
                
                # Ajouter une vérification des dimensions
                print(f"    Debug: s_t shape={s_t.shape}, a_t shape={a_t.shape}")
                
                a_tgt = mapper(s_t, a_t).detach().cpu().numpy().flatten()
                
                # 5. Step
                obs, _, done, _, info = env.step(a_tgt)
            
            if info.get('target_reached', False):
                successes += 1
            
            if (ep + 1) % 20 == 0:
                print(f"    Episode {ep+1}: {100*successes/(ep+1):.1f}%")
        
        rate = 100 * successes / n_episodes
        print(f"    Final success rate: {rate:.1f}%")
        env.close()
        return rate


# ================================================================
# Main
# ================================================================

def main():
    # Forcer l'utilisation du CPU pour éviter les erreurs de mémoire GPU
    DEVICE = "cpu"
    print(f"Device: {DEVICE}")
    print("Note: Using CPU to avoid CUDA out of memory errors")
    
    # Nettoyer la mémoire
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer = JointTransferTrainer(latent_dim=16, device=DEVICE)
    
    # Phase 1: Trajectoires aléatoires (paramètres réduits)
    data = trainer.collect_random_paired_trajectories(n_pairs=10000, seq_len=30)
    
    # Phase 2: VAE
    trainer.train_vae(data, epochs=150)
    
    # Phase 3: Politiques PPO (temps réduit)
    trainer.train_latent_policies(total_timesteps=5_000_000)
    
    # Phase 4: Mappers
    trainer.train_mappers(data, epochs=200)
    
    # Sauvegarde
    trainer.save()
    
    # Tests de transfert bidirectionnel (nombre d'épisodes réduit)
    print("\n" + "="*60)
    print("Transfer Results")
    print("="*60)
    
    rate_2to3 = trainer.test_transfer_2to3(n_episodes=10)  # Test avec peu d'épisodes d'abord
    rate_3to2 = trainer.test_transfer_3to2(n_episodes=10)
    
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"  Transfer 2DoF → 3DoF: {rate_2to3:.1f}%")
    print(f"  Transfer 3DoF → 2DoF: {rate_3to2:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
