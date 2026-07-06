import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ajouter le chemin du projet si nécessaire
import sys
sys.path.append('.')

from envs.env_pushball_2dof_rec import PushBallEnv_2dof
from envs.arm_2dof import Arm2DoF
from direct_method.mapper_models import ARM_OBS_2DOF, ARM_OBS_3DOF, MAX_REACH


def create_2dof_state(theta1, theta2, dtheta1=0.0, dtheta2=0.0):
    """
    Crée un état normalisé pour le bras 2DoF à partir des angles en radians.
    """
    eff = np.array([
        1.5 * np.cos(theta1) + 1.5 * np.cos(theta1 + theta2),
        1.5 * np.sin(theta1) + 1.5 * np.sin(theta1 + theta2)
    ], dtype=np.float32)
    
    return np.array([
        theta1 / np.pi,
        theta2 / np.pi,
        dtheta1 / 2.0,
        dtheta2 / 2.0,
        eff[0] / 3.0,
        eff[1] / 3.0
    ], dtype=np.float32)


def compute_reconstruction_error(env, theta1, theta2):
    """
    Calcule l'erreur de reconstruction pour un état donné (theta1, theta2).
    Retourne les différentes métriques d'erreur.
    """
    # Créer l'état normalisé
    arm_state = create_2dof_state(theta1, theta2)
    action = np.array([0.0, 0.0], dtype=np.float32)  # Action nulle
    
    # Vérifier que les mappers sont chargés
    if not env._mappers_ready:
        print("Mappers non chargés !")
        return None, None, None, None
    
    # Exécuter la round-trip
    s2 = torch.tensor(arm_state, dtype=torch.float32).unsqueeze(0)
    a2 = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    
    # State round-trip: 2 → 3 → 2
    s3 = env._sm_2to3(s2)
    s2_recon = env._sm_3to2(s3)
    
    s2_recon_np = s2_recon.squeeze(0).cpu().numpy()
    
    # Calcul des similarités
    geometric_rec = env._geometric_similarity_2dof(arm_state, s2_recon_np)
    effector_rec = env._effector_similarity_2dof(arm_state, s2_recon_np)
    
    # Erreurs (1 - similarité)
    geometric_error = 1.0 - geometric_rec
    effector_error = 1.0 - effector_rec
    
    # Erreur d'état combinée selon la formule de l'environnement
    state_sim = env.eff_ratio * effector_rec + (1.0 - env.eff_ratio) * geometric_rec
    state_error = 1.0 - state_sim
    
    # Erreur sur les angles (pour une visualisation plus directe)
    # On dé-normalise les angles
    theta1_recon = s2_recon_np[0] * np.pi
    theta2_recon = s2_recon_np[1] * np.pi
    angle_error = np.sqrt((theta1 - theta1_recon)**2 + (theta2 - theta2_recon)**2)
    
    # Erreur sur l'end-effector en position
    eff_recon = np.array([
        1.5 * np.cos(theta1_recon) + 1.5 * np.cos(theta1_recon + theta2_recon),
        1.5 * np.sin(theta1_recon) + 1.5 * np.sin(theta1_recon + theta2_recon)
    ])
    eff_original = np.array([
        1.5 * np.cos(theta1) + 1.5 * np.cos(theta1 + theta2),
        1.5 * np.sin(theta1) + 1.5 * np.sin(theta1 + theta2)
    ])
    effector_position_error = np.linalg.norm(eff_original - eff_recon)
    
    return {
        'geometric_error': geometric_error,
        'effector_error': effector_error,
        'state_error': state_error,
        'angle_error': angle_error,
        'effector_position_error': effector_position_error,
        'theta1_recon': theta1_recon,
        'theta2_recon': theta2_recon
    }


def plot_reconstruction_map(env, resolution=50, save_path=None):
    """
    Génère une carte 2D de l'erreur de reconstruction en fonction de theta1 et theta2.
    
    Args:
        env: L'environnement PushBallEnv_2dof avec les mappers chargés
        resolution: Nombre de points par dimension
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    # Définir la grille d'angles
    theta1_range = np.linspace(-np.pi, np.pi, resolution)
    theta2_range = np.linspace(-np.pi, np.pi, resolution)
    theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)
    
    # Initialiser les matrices d'erreur
    geo_error_grid = np.zeros_like(theta1_grid)
    eff_error_grid = np.zeros_like(theta1_grid)
    state_error_grid = np.zeros_like(theta1_grid)
    angle_error_grid = np.zeros_like(theta1_grid)
    pos_error_grid = np.zeros_like(theta1_grid)
    
    # Calculer l'erreur pour chaque point
    print(f"Calcul des erreurs sur une grille {resolution}x{resolution}...")
    for i in range(resolution):
        for j in range(resolution):
            theta1 = theta1_grid[i, j]
            theta2 = theta2_grid[i, j]
            result = compute_reconstruction_error(env, theta1, theta2)
            
            if result is not None:
                geo_error_grid[i, j] = result['geometric_error']
                eff_error_grid[i, j] = result['effector_error']
                state_error_grid[i, j] = result['state_error']
                angle_error_grid[i, j] = result['angle_error']
                pos_error_grid[i, j] = result['effector_position_error']
    
    # Créer la figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Carte d'erreur de reconstruction 2DoF en fonction de q1 et q2", fontsize=16)
    
    # Définir les titres et les couleurs
    plots = [
        (geo_error_grid, 'Erreur géométrique\n(1 - geometric_rec)', 'viridis'),
        (eff_error_grid, 'Erreur effecteur\n(1 - effector_rec)', 'viridis'),
        (state_error_grid, 'Erreur d\'état combinée\n(1 - state_sim)', 'viridis'),
        (angle_error_grid, 'Erreur angulaire\n(radians)', 'plasma'),
        (pos_error_grid, 'Erreur de position de l\'EE\n(mètres)', 'plasma'),
    ]
    
    for idx, (data, title, cmap) in enumerate(plots):
        ax = axes[idx // 3, idx % 3]
        im = ax.imshow(data, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                      origin='lower', cmap=cmap, aspect='auto')
        ax.set_xlabel('θ1 (rad)')
        ax.set_ylabel('θ2 (rad)')
        ax.set_title(title)
        
        # Ajouter une colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Ajouter un sous-graphe vide ou un résumé
    ax = axes[1, 2]
    ax.axis('off')
    
    # Ajouter des informations sur les paramètres
    info_text = (
        f"Paramètres de l'environnement:\n"
        f"w_recon = {env.w_recon}\n"
        f"eff_ratio = {env.eff_ratio}\n"
        f"Mappers chargés: {env._mappers_ready}\n\n"
        f"state_sim = eff_ratio*effector_rec + (1-eff_ratio)*geometric_rec\n"
        f"recon_term = 0.5*state_sim + 0.5*velocity_rec\n"
        f"recon_reward = w_recon*recon_term"
    )
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()
    return fig


def plot_error_distribution(env, n_samples=10000, save_path=None):
    """
    Visualise la distribution des erreurs de reconstruction sur un échantillon aléatoire.
    """
    # Générer des états aléatoires
    theta1_samples = np.random.uniform(-np.pi, np.pi, n_samples)
    theta2_samples = np.random.uniform(-np.pi, np.pi, n_samples)
    
    errors = {
        'geometric': [],
        'effector': [],
        'state': [],
        'angle': [],
        'position': []
    }
    
    print(f"Échantillonnage de {n_samples} états aléatoires...")
    for i in range(n_samples):
        result = compute_reconstruction_error(env, theta1_samples[i], theta2_samples[i])
        if result is not None:
            errors['geometric'].append(result['geometric_error'])
            errors['effector'].append(result['effector_error'])
            errors['state'].append(result['state_error'])
            errors['angle'].append(result['angle_error'])
            errors['position'].append(result['effector_position_error'])
    
    # Créer la figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Distribution des erreurs de reconstruction (2DoF)", fontsize=14)
    
    for idx, (key, values) in enumerate(errors.items()):
        ax = axes[idx // 3, idx % 3]
        ax.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Erreur')
        ax.set_ylabel('Fréquence')
        ax.set_title(f"{key.capitalize()}\nμ={np.mean(values):.4f}, σ={np.std(values):.4f}")
        ax.axvline(np.mean(values), color='red', linestyle='--', label='Moyenne')
        ax.legend()
    
    ax = axes[1, 2]
    ax.axis('off')
    
    # Statistiques résumées
    stats_text = "Statistiques résumées:\n\n"
    for key, values in errors.items():
        stats_text += f"{key}:\n"
        stats_text += f"  min: {np.min(values):.4f}\n"
        stats_text += f"  max: {np.max(values):.4f}\n"
        stats_text += f"  mean: {np.mean(values):.4f}\n"
        stats_text += f"  std: {np.std(values):.4f}\n\n"
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()
    return fig, errors


# ============================================================================
# Script principal
# ============================================================================
if __name__ == "__main__":
    # Chemins des mappers
    state_mapper_2to3_path = './direct_method/runs/run_01/models/state_mapper_r1_to_r2.pt'
    state_mapper_3to2_path = './direct_method/runs/run_01/models/state_mapper_r2_to_r1.pt'
    action_mapper_2to3_path = './direct_method/runs/run_02/models/action_mapper_r1_to_r2.pt'
    action_mapper_3to2_path = './direct_method/runs/run_02/models/action_mapper_r2_to_r1.pt'
    
    # Créer l'environnement
    print("Chargement de l'environnement...")
    env = PushBallEnv_2dof(
        render_mode=None,
        state_mapper_2to3_path=state_mapper_2to3_path,
        state_mapper_3to2_path=state_mapper_3to2_path,
        action_mapper_2to3_path=action_mapper_2to3_path,
        action_mapper_3to2_path=action_mapper_3to2_path,
        w_recon=1.0,
        eff_ratio=0.5,
    )
    
    # Vérifier que les mappers sont chargés
    if not env._mappers_ready:
        print("ERREUR: Les mappers ne sont pas chargés. Vérifiez les chemins.")
        print("Mappers chargés:")
        print(f"  _sm_2to3: {env._sm_2to3 is not None}")
        print(f"  _sm_3to2: {env._sm_3to2 is not None}")
        print(f"  _am_2to3: {env._am_2to3 is not None}")
        print(f"  _am_3to2: {env._am_3to2 is not None}")
        exit(1)
    
    print("Environnement chargé avec succès.")
    print(f"w_recon = {env.w_recon}, eff_ratio = {env.eff_ratio}")
    
    # Créer le dossier de sortie
    output_dir = "./reconstruction_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Carte 2D de l'erreur de reconstruction
    print("\n--- Génération de la carte 2D ---")
    fig1 = plot_reconstruction_map(
        env, 
        resolution=100, 
        save_path=os.path.join(output_dir, "reconstruction_map_2dof.png")
    )
    
    # 2. Distribution des erreurs
    print("\n--- Analyse de la distribution des erreurs ---")
    fig2, errors = plot_error_distribution(
        env,
        n_samples=50000,
        save_path=os.path.join(output_dir, "error_distribution_2dof.png")
    )
    
    # 3. Analyse spécifique pour certaines configurations d'angles
    print("\n--- Analyse de configurations spécifiques ---")
    configs = [
        (0.0, 0.0, "Bras tendu vers l'avant"),
        (0.0, np.pi/2, "Coude à 90°"),
        (np.pi/4, np.pi/4, "Angles positifs"),
        (-np.pi/4, -np.pi/4, "Angles négatifs"),
        (np.pi/2, 0.0, "Premier angle à 90°"),
        (0.0, np.pi, "Coude replié"),
    ]
    
    print("\nErreurs pour des configurations spécifiques:")
    print("-" * 70)
    print(f"{'Config':<25} {'Géo':<10} {'Eff':<10} {'State':<10} {'Angle':<10}")
    print("-" * 70)
    
    for theta1, theta2, desc in configs:
        result = compute_reconstruction_error(env, theta1, theta2)
        if result:
            print(f"{desc:<25} {result['geometric_error']:<10.4f} {result['effector_error']:<10.4f} {result['state_error']:<10.4f} {result['angle_error']:<10.4f}")
    
    print("\nAnalyse terminée.")