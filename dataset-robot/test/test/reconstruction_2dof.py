import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# Ajouter le chemin du projet si nécessaire
import sys
sys.path.append('.')

from envs.env_pushball_2dof_rec import PushBallEnv_2dof


def create_2dof_state(theta1, theta2, dtheta1=0.0, dtheta2=0.0):
    """
    Crée un état normalisé pour le bras 2DoF à partir des angles en radians.
    """
    _L1_2 = _L2_2 = 1.5
    max_reach = 3.0
    
    eff = np.array([
        _L1_2 * np.cos(theta1) + _L2_2 * np.cos(theta1 + theta2),
        _L1_2 * np.sin(theta1) + _L2_2 * np.sin(theta1 + theta2)
    ], dtype=np.float32)
    
    return np.array([
        theta1 / np.pi,
        theta2 / np.pi,
        dtheta1 / 2.0,
        dtheta2 / 2.0,
        eff[0] / max_reach,
        eff[1] / max_reach
    ], dtype=np.float32)


def compute_reconstruction_error(env, theta1, theta2):
    """
    Calcule l'erreur de reconstruction pour un état donné (theta1, theta2).
    Retourne l'erreur géométrique et l'erreur de position de l'effecteur.
    """
    arm_state = create_2dof_state(theta1, theta2)
    action = np.array([0.0, 0.0], dtype=np.float32)
    
    if not env._mappers_ready:
        return None, None
    
    # Round-trip: 2 → 3 → 2
    s2 = torch.tensor(arm_state, dtype=torch.float32).unsqueeze(0)
    a2 = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    
    s3 = env._sm_2to3(s2)
    s2_recon = env._sm_3to2(s3)
    
    s2_recon_np = s2_recon.squeeze(0).cpu().numpy()
    
    # Calcul des similarités (comme dans l'environnement)
    # Utilisation des fonctions de l'environnement
    geometric_rec = env._geometric_similarity_2dof(arm_state, s2_recon_np)
    effector_rec = env._effector_similarity_2dof(arm_state, s2_recon_np)
    
    # Erreurs (1 - similarité)
    geometric_error = 1.0 - geometric_rec
    effector_position_error = 1.0 - effector_rec
    
    return geometric_error, effector_position_error


def create_3d_error_plots(env, resolution=80, save_path=None):
    """
    Crée deux graphiques 3D montrant l'erreur de reconstruction en fonction de q1 et q2.
    
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
    
    # Calculer l'erreur pour chaque point
    print(f"Calcul des erreurs sur une grille {resolution}x{resolution}...")
    total_points = resolution * resolution
    count = 0
    
    for i in range(resolution):
        for j in range(resolution):
            theta1 = theta1_grid[i, j]
            theta2 = theta2_grid[i, j]
            geo_err, eff_err = compute_reconstruction_error(env, theta1, theta2)
            
            if geo_err is not None:
                geo_error_grid[i, j] = geo_err
                eff_error_grid[i, j] = eff_err
            
            count += 1
            if count % 1000 == 0:
                print(f"  Progression: {count}/{total_points}")
    
    # Créer la figure avec deux sous-graphiques 3D
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Carte 3D de l'erreur de reconstruction 2DoF", fontsize=16, fontweight='bold')
    
    # Premier graphique : Erreur géométrique
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(theta1_grid, theta2_grid, geo_error_grid, 
                             cmap='viridis', linewidth=0, antialiased=True, 
                             alpha=0.9, rstride=2, cstride=2)
    ax1.set_xlabel('θ1 (rad)', fontsize=12, labelpad=10)
    ax1.set_ylabel('θ2 (rad)', fontsize=12, labelpad=10)
    ax1.set_zlabel('Erreur géométrique\n(1 - geometric_rec)', fontsize=12, labelpad=10)
    ax1.set_title('Erreur géométrique\n(Posture globale du bras)', fontsize=13)
    ax1.view_init(elev=30, azim=-45)
    
    # Ajouter une colorbar
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=15, pad=0.1)
    cbar1.set_label('Erreur', fontsize=11)
    
    # Deuxième graphique : Erreur de position de l'effecteur
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(theta1_grid, theta2_grid, eff_error_grid, 
                             cmap='plasma', linewidth=0, antialiased=True, 
                             alpha=0.9, rstride=2, cstride=2)
    ax2.set_xlabel('θ1 (rad)', fontsize=12, labelpad=10)
    ax2.set_ylabel('θ2 (rad)', fontsize=12, labelpad=10)
    ax2.set_zlabel('Erreur effecteur\n(1 - effector_rec)', fontsize=12, labelpad=10)
    ax2.set_title('Erreur de position de l\'effecteur\n(End-effector uniquement)', fontsize=13)
    ax2.view_init(elev=30, azim=-45)
    
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=15, pad=0.1)
    cbar2.set_label('Erreur', fontsize=11)
    
    # Ajuster la disposition
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()
    return fig, geo_error_grid, eff_error_grid


def create_combined_3d_plot(env, resolution=80, save_path=None):
    """
    Crée un graphique 3D combiné avec les deux surfaces.
    """
    theta1_range = np.linspace(-np.pi, np.pi, resolution)
    theta2_range = np.linspace(-np.pi, np.pi, resolution)
    theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)
    
    geo_error_grid = np.zeros_like(theta1_grid)
    eff_error_grid = np.zeros_like(theta1_grid)
    
    print(f"Calcul des erreurs sur une grille {resolution}x{resolution}...")
    for i in range(resolution):
        for j in range(resolution):
            theta1 = theta1_grid[i, j]
            theta2 = theta2_grid[i, j]
            geo_err, eff_err = compute_reconstruction_error(env, theta1, theta2)
            if geo_err is not None:
                geo_error_grid[i, j] = geo_err
                eff_error_grid[i, j] = eff_err
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface d'erreur géométrique
    surf1 = ax.plot_surface(theta1_grid, theta2_grid, geo_error_grid, 
                            cmap='viridis', linewidth=0, antialiased=True,
                            alpha=0.7, label='Erreur géométrique')
    
    # Surface d'erreur de l'effecteur
    surf2 = ax.plot_surface(theta1_grid, theta2_grid, eff_error_grid, 
                            cmap='plasma', linewidth=0, antialiased=True,
                            alpha=0.7, label='Erreur effecteur')
    
    ax.set_xlabel('θ1 (rad)', fontsize=13, labelpad=12)
    ax.set_ylabel('θ2 (rad)', fontsize=13, labelpad=12)
    ax.set_zlabel('Erreur', fontsize=13, labelpad=12)
    ax.set_title('Comparaison des erreurs de reconstruction 2DoF', fontsize=15, fontweight='bold')
    ax.view_init(elev=25, azim=-55)
    
    # Ajouter une légende avec des patches personnalisés
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='viridis', alpha=0.7, label='Erreur géométrique'),
        Patch(facecolor='plasma', alpha=0.7, label='Erreur effecteur')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()
    return fig


def create_contour_plots(env, resolution=80, save_path=None):
    """
    Crée des graphiques en contour 2D des erreurs.
    """
    theta1_range = np.linspace(-np.pi, np.pi, resolution)
    theta2_range = np.linspace(-np.pi, np.pi, resolution)
    theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)
    
    geo_error_grid = np.zeros_like(theta1_grid)
    eff_error_grid = np.zeros_like(theta1_grid)
    
    print(f"Calcul des erreurs sur une grille {resolution}x{resolution}...")
    for i in range(resolution):
        for j in range(resolution):
            theta1 = theta1_grid[i, j]
            theta2 = theta2_grid[i, j]
            geo_err, eff_err = compute_reconstruction_error(env, theta1, theta2)
            if geo_err is not None:
                geo_error_grid[i, j] = geo_err
                eff_error_grid[i, j] = eff_err
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cartes de contour de l'erreur de reconstruction 2DoF", fontsize=14, fontweight='bold')
    
    # Premier contour : Erreur géométrique
    contour1 = axes[0].contourf(theta1_grid, theta2_grid, geo_error_grid, 
                               levels=20, cmap='viridis')
    axes[0].set_xlabel('θ1 (rad)')
    axes[0].set_ylabel('θ2 (rad)')
    axes[0].set_title('Erreur géométrique\n(1 - geometric_rec)')
    axes[0].grid(True, alpha=0.3)
    cbar1 = fig.colorbar(contour1, ax=axes[0])
    cbar1.set_label('Erreur')
    
    # Deuxième contour : Erreur de l'effecteur
    contour2 = axes[1].contourf(theta1_grid, theta2_grid, eff_error_grid, 
                                levels=20, cmap='plasma')
    axes[1].set_xlabel('θ1 (rad)')
    axes[1].set_ylabel('θ2 (rad)')
    axes[1].set_title('Erreur effecteur\n(1 - effector_rec)')
    axes[1].grid(True, alpha=0.3)
    cbar2 = fig.colorbar(contour2, ax=axes[1])
    cbar2.set_label('Erreur')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()
    return fig


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
    
    # 1. Deux graphiques 3D séparés
    print("\n--- Génération des graphiques 3D ---")
    fig1, geo_error, eff_error = create_3d_error_plots(
        env, 
        resolution=80,
        save_path=os.path.join(output_dir, "reconstruction_3d_separate.png")
    )
    
    # 2. Graphique 3D combiné
    print("\n--- Génération du graphique 3D combiné ---")
    fig2 = create_combined_3d_plot(
        env,
        resolution=80,
        save_path=os.path.join(output_dir, "reconstruction_3d_combined.png")
    )
    
    # 3. Cartes de contour 2D
    print("\n--- Génération des cartes de contour ---")
    fig3 = create_contour_plots(
        env,
        resolution=80,
        save_path=os.path.join(output_dir, "reconstruction_contours.png")
    )
    
    # 4. Analyse des statistiques
    print("\n--- Statistiques des erreurs ---")
    print("-" * 50)
    print(f"Erreur géométrique:")
    print(f"  Min: {np.min(geo_error):.6f}")
    print(f"  Max: {np.max(geo_error):.6f}")
    print(f"  Moyenne: {np.mean(geo_error):.6f}")
    print(f"  Écart-type: {np.std(geo_error):.6f}")
    print()
    print(f"Erreur effecteur:")
    print(f"  Min: {np.min(eff_error):.6f}")
    print(f"  Max: {np.max(eff_error):.6f}")
    print(f"  Moyenne: {np.mean(eff_error):.6f}")
    print(f"  Écart-type: {np.std(eff_error):.6f}")
    print()
    print(f"Différence moyenne entre les deux erreurs: {np.mean(eff_error - geo_error):.6f}")
    
    print("\nAnalyse terminée.")