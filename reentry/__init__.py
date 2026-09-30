"""ALAKAZAM: ALuminium Ablation Kinetics And Z-resolved Atmospheric injection Model.

The model package: atmosphere (exponential placeholder and NRLMSIS), vehicle
and entry state, trajectory integration, aerodynamic heating, the per-band
thermal and ablation model, and emissivity.
"""
from .atmosphere import ExponentialAtmosphere, MSISAtmosphere
from .trajectory import allen_eggers, eom, integrate
from .vehicle import EntryState, Vehicle

__all__ = [
    "ExponentialAtmosphere",
    "MSISAtmosphere",
    "Vehicle",
    "EntryState",
    "integrate",
    "eom",
    "allen_eggers",
]
