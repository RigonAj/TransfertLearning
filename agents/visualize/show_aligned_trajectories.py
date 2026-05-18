import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_continuous_reaching_2dof import Arm2DoFPersistentEnv
from envs.env_continuous_reaching_3dof import Arm3DoFPersistentEnv


run_id_2dof = 1
run_id_3dof = 1

MODEL_2DOF   = f"models/ppo_reach_2dof_{run_id_2dof}/best_model.zip"
VECNORM_2DOF = f"models/ppo_reach_2dof_{run_id_2dof}/vec_normalize.pkl"
MODEL_3DOF   = f"models/ppo_reach_3dof_{run_id_3dof}/best_model.zip"
VECNORM_3DOF = f"models/ppo_reach_3dof_{run_id_3dof}/vec_normalize.pkl"

COL_LINK1   = "#E05A3A"
COL_LINK2   = "#3A8FE0"
COL_LINK3   = "#9B5DE5"
COL_EFF     = "#FFFFFF"
COL_EFF_OK  = "#55DD88"
COL_TARGET  = "#F7B731"


def load_policy(model_path, vecnorm_path, env_factory, device="cpu"):
    model = PPO.load(model_path, device=device, custom_objects={
        "learning_rate": 3e-4,
        "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2,
    })
    venv = DummyVecEnv([env_factory])
    vec_norm = VecNormalize.load(vecnorm_path, venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm, venv

def predict(model, vec_norm, raw_obs):
    obs_norm = vec_norm.normalize_obs(raw_obs.reshape(1, -1))
    action, _ = model.predict(obs_norm, deterministic=True)
    return action[0]

def override_target(env, target):
    """Change la cible et réinitialise proprement le flag persistant."""
    # Réinitialisation explicite du flag (sécurité)
    if hasattr(env, '_target_reached'):
        env._target_reached = False
    # Utiliser la méthode set_target si elle existe (elle fait déjà le travail)
    if hasattr(env, 'set_target'):
        env.set_target(target)
    else:
        env.target = target.copy().astype(np.float32)
        if hasattr(env, 'theta3'):
            ee = env.forward_kinematics(env.theta1, env.theta2, env.theta3)
        else:
            ee = env.forward_kinematics(env.theta1, env.theta2)
        env.prev_dist = float(np.linalg.norm(ee - target))

def fk2(env, t1=None, t2=None):
    t1 = t1 if t1 is not None else env.theta1
    t2 = t2 if t2 is not None else env.theta2
    j1 = np.array([env.l1 * np.cos(t1), env.l1 * np.sin(t1)])
    eff = env.forward_kinematics(t1, t2)
    return np.zeros(2), j1, eff

def fk3(env, t1=None, t2=None, t3=None):
    t1 = t1 if t1 is not None else env.theta1
    t2 = t2 if t2 is not None else env.theta2
    t3 = t3 if t3 is not None else env.theta3
    j1 = np.array([env.l1 * np.cos(t1), env.l1 * np.sin(t1)])
    j2 = j1 + np.array([env.l2 * np.cos(t1 + t2), env.l2 * np.sin(t1 + t2)])
    eff = env.forward_kinematics(t1, t2, t3)
    return np.zeros(2), j1, j2, eff

def draw_2dof(ax, env, targets, target_idx, step_count, success):
    ax.cla()
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta)*env.max_reach, np.sin(theta)*env.max_reach,
            color="#FFFFFF", lw=0.6, ls="--", alpha=0.3)

    for i, tgt in enumerate(targets):
        if i == target_idx:
            ax.plot(tgt[0], tgt[1], 'o', color=COL_TARGET, markersize=16,
                    markeredgecolor="#000", markeredgewidth=0.5, zorder=5)

    o, j1, eff = fk2(env)
    ax.plot([o[0],  j1[0]],  [o[1],  j1[1]],  '-', color=COL_LINK1, lw=6, solid_capstyle='round')
    ax.plot([j1[0], eff[0]], [j1[1], eff[1]], '-', color=COL_LINK2, lw=6, solid_capstyle='round')
    dist = np.linalg.norm(eff - env.target)
    ec = COL_EFF_OK if success or dist < env.epsilon else COL_EFF
    ax.plot(*eff, 'o', color=ec, markersize=11,
            markeredgecolor="#222", markeredgewidth=1.2, zorder=7)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"2-DoF  |  step {step_count:3d}  |  cible {target_idx+1}/{len(targets)}\n"
                 f"θ1={np.degrees(env.theta1):+.0f}°  θ2={np.degrees(env.theta2):+.0f}°  d={dist:.3f} m",
                 color="black", fontsize=9, pad=6)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def draw_3dof(ax, env, targets, target_idx, step_count, success):
    ax.cla()
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta)*env.max_reach, np.sin(theta)*env.max_reach,
            color="#FFFFFF", lw=0.6, ls="--", alpha=0.3)

    for i, tgt in enumerate(targets):
        if i == target_idx:
            ax.plot(tgt[0], tgt[1], 'o', color=COL_TARGET, markersize=16,
                    markeredgecolor="#000", markeredgewidth=0.5, zorder=5)

    o, j1, j2, eff = fk3(env)
    ax.plot([o[0],  j1[0]],  [o[1],  j1[1]],  '-', color=COL_LINK1, lw=6, solid_capstyle='round')
    ax.plot([j1[0], j2[0]], [j1[1], j2[1]], '-', color=COL_LINK2, lw=6, solid_capstyle='round')
    ax.plot([j2[0], eff[0]], [j2[1], eff[1]], '-', color=COL_LINK3, lw=6, solid_capstyle='round')
    dist = np.linalg.norm(eff - env.target)
    ec = COL_EFF_OK if success or dist < env.epsilon else COL_EFF
    ax.plot(*eff, 'o', color=ec, markersize=11,
            markeredgecolor="#222", markeredgewidth=1.2, zorder=7)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"3-DoF  |  step {step_count:3d}  |  cible {target_idx+1}/{len(targets)}\n"
                 f"θ1={np.degrees(env.theta1):+.0f}°  θ2={np.degrees(env.theta2):+.0f}°  θ3={np.degrees(env.theta3):+.0f}°  d={dist:.3f} m",
                 color="black", fontsize=9, pad=6)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def generate_target_sequence(seed, n_targets, max_reach, scale=0.7, min_dist=0.1):
    """Génère des cibles avec une distance minimale entre cibles consécutives."""
    rng = np.random.RandomState(seed)
    targets = []
    for _ in range(n_targets):
        while True:
            r = rng.uniform(0.0, max_reach * scale)
            angle = rng.uniform(-np.pi, np.pi)
            new_target = np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
            if len(targets) == 0 or np.linalg.norm(new_target - targets[-1]) >= min_dist:
                targets.append(new_target)
                break
    return targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, default=1, help="Seed pour la séquence de cibles (paire)")
    parser.add_argument("--delay", type=float, default=0.04)
    parser.add_argument("--n_targets", type=int, default=10, help="Nombre de cibles dans la séquence")
    parser.add_argument("--steps_per_target", type=int, default=50, help="Steps max par cible")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Distance minimale entre cibles consécutives")
    args = parser.parse_args()

    print(f"\nPaire {args.pair}\n")

    # Charger les politiques
    def make_2dof(): return Monitor(Arm2DoFPersistentEnv(render_mode=None))
    def make_3dof(): return Monitor(Arm3DoFPersistentEnv(render_mode=None))
    model_2, vn_2, venv_2 = load_policy(MODEL_2DOF, VECNORM_2DOF, make_2dof)
    model_3, vn_3, venv_3 = load_policy(MODEL_3DOF, VECNORM_3DOF, make_3dof)

    # Créer les environnements visuels
    env_2 = Arm2DoFPersistentEnv(render_mode="human")
    env_3 = Arm3DoFPersistentEnv(render_mode="human")

    # Générer les cibles
    max_reach = min(env_2.max_reach, env_3.max_reach)
    targets = generate_target_sequence(args.pair, args.n_targets, max_reach, scale=0.7, min_dist=args.min_dist)
    n_targets = len(targets)

    # Réinitialiser les bras avec la même seed
    seed = args.pair
    env_2.reset(seed=seed)
    env_3.reset(seed=seed)
    # Forcer les angles à 0 pour partir de la même position (optionnel)
    env_2.theta1 = 0.0; env_2.theta2 = 0.0
    env_3.theta1 = 0.0; env_3.theta2 = 0.0; env_3.theta3 = 0.0
    override_target(env_2, targets[0])
    override_target(env_3, targets[0])

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Trajectoire alignée 2-DoF ↔ 3-DoF  |  paire {args.pair}", color="black", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.ion()
    plt.show()

    target_idx = 0
    step_in_tgt = 0
    step_count = 0
    success_2 = False
    success_3 = False

    total_steps = n_targets * args.steps_per_target

    for global_step in range(total_steps):
        if not plt.fignum_exists(fig.number):
            print("Fermeture fenêtre")
            break

        # Changement de cible si nécessaire
        if step_in_tgt >= args.steps_per_target and target_idx < n_targets-1:
            if target_idx > 0:
                status = "valid" if (success_2 and success_3) else "invalid"
                print(f"  Segment {target_idx} : {status}  (cible {target_idx} → {target_idx+1})")
            target_idx += 1
            step_in_tgt = 0
            override_target(env_2, targets[target_idx])
            override_target(env_3, targets[target_idx])
            success_2 = False
            success_3 = False

        # Actions
        act_2 = predict(model_2, vn_2, env_2._get_obs())
        act_3 = predict(model_3, vn_3, env_3._get_obs())
        obs_2, _, term2, trunc2, info2 = env_2.step(act_2)
        obs_3, _, term3, trunc3, info3 = env_3.step(act_3)

        if info2.get("target_reached", False):
            success_2 = True
        if info3.get("target_reached", False):
            success_3 = True

        # Réinitialisation si terminaison (normalement pas avec persistent)
        if term2 or trunc2:
            env_2.reset()
            override_target(env_2, targets[target_idx])
        if term3 or trunc3:
            env_3.reset()
            override_target(env_3, targets[target_idx])

        step_count += 1
        step_in_tgt += 1

        draw_2dof(ax2, env_2, targets, target_idx, step_count, success_2)
        draw_3dof(ax3, env_3, targets, target_idx, step_count, success_3)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(args.delay)

        '''
        if step_count % (args.steps_per_target * 5) == 0:
            d2 = np.linalg.norm(env_2.forward_kinematics(env_2.theta1, env_2.theta2) - targets[target_idx])
            d3 = np.linalg.norm(env_3.forward_kinematics(env_3.theta1, env_3.theta2, env_3.theta3) - targets[target_idx])
            print(f"  step {step_count:4d} | target {target_idx+1:3d}/{n_targets} | dist 2={d2:.3f}  3={d3:.3f}")
        '''

    # Dernier segment
    if n_targets > 1:
        status = "valid" if (success_2 and success_3) else "invalid"
        print(f"  Segment {n_targets-1} : {status}  (cible {n_targets-1} → {n_targets})")

    plt.ioff()
    plt.show()
    env_2.close()
    env_3.close()
    venv_2.close()
    venv_3.close()

if __name__ == "__main__":
    main()
