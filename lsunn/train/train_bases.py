"""
Train LS-UNN VAE Bases for PushBall task.

Usage:
    python -m lsunn.train_bases
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.bases_vae import main

if __name__ == "__main__":
    main()
