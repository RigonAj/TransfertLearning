"""
IK analytique pour génération de trajectoires primitives de reaching.
N'affecte pas les environnements Gym - utilisée uniquement par trajectories.py.
"""

import numpy as np
import torch


def ik_2dof(
    target_xy: np.ndarray, 
    l1: float = 1.5, 
    l2: float = 1.5
) -> np.ndarray:
    """
    Résout l'IK pour un bras 2DoF planaire (loi des cosinus).
    
    Args:
        target_xy: (x, y) position cible de l'effecteur
        l1, l2: longueurs des segments
        
    Returns:
        (theta1, theta2) en radians
    """
    x, y = target_xy
    r = np.sqrt(x**2 + y**2)
    r = np.clip(r, 0.001, l1 + l2 - 0.001)  # éviter singularités
    
    # Loi des cosinus pour theta2
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    
    # theta1 = atan2(y,x) - atan2(l2*sin(theta2), l1 + l2*cos(theta2))
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    
    return np.array([theta1, theta2])


def ik_3dof(
    target_xy: np.ndarray,
    l1: float = 1.0,
    l2: float = 1.0,
    l3: float = 1.0,
    theta_total: float = np.pi / 3  # orientation fixe du dernier segment
) -> np.ndarray:
    """
    Résout l'IK pour un bras 3DoF planaire avec orientation fixée.
    
    Contrainte: theta1 + theta2 + theta3 = theta_total (constant)
    Ramène à un problème 2DoF sur les deux premiers segments.
    
    Args:
        target_xy: (x, y) position cible de l'effecteur
        l1, l2, l3: longueurs des segments
        theta_total: orientation constante du dernier segment (rad)
        
    Returns:
        (theta1, theta2, theta3) en radians
    """
    x, y = target_xy
    
    # Position du joint 2 = target - l3 * orientation fixée
    j2_x = x - l3 * np.cos(theta_total)
    j2_y = y - l3 * np.sin(theta_total)
    
    # IK 2DoF pour les deux premiers segments vers j2
    theta1, theta2 = ik_2dof(np.array([j2_x, j2_y]), l1, l2)
    theta3 = theta_total - theta1 - theta2
    
    return np.array([theta1, theta2, theta3])


# Version PyTorch pour batch processing
def ik_2dof_torch(
    target_xy: torch.Tensor,
    l1: float = 1.5,
    l2: float = 1.5
) -> torch.Tensor:
    """Version batch PyTorch de ik_2dof."""
    x, y = target_xy[:, 0], target_xy[:, 1]
    r = torch.sqrt(x**2 + y**2)
    r = torch.clamp(r, 0.001, l1 + l2 - 0.001)
    
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = torch.clamp(cos_theta2, -1.0, 1.0)
    theta2 = torch.acos(cos_theta2)
    
    theta1 = torch.atan2(y, x) - torch.atan2(l2 * torch.sin(theta2), l1 + l2 * torch.cos(theta2))
    
    return torch.stack([theta1, theta2], dim=1)


def ik_3dof_torch(
    target_xy: torch.Tensor,
    l1: float = 1.0,
    l2: float = 1.0,
    l3: float = 1.0,
    theta_total: float = np.pi / 3
) -> torch.Tensor:
    """Version batch PyTorch de ik_3dof."""
    x, y = target_xy[:, 0], target_xy[:, 1]
    
    j2_x = x - l3 * np.cos(theta_total)
    j2_y = y - l3 * np.sin(theta_total)
    j2_xy = torch.stack([j2_x, j2_y], dim=1)
    
    theta1, theta2 = ik_2dof_torch(j2_xy, l1, l2)
    theta3 = theta_total - theta1 - theta2
    
    return torch.stack([theta1, theta2, theta3], dim=1)