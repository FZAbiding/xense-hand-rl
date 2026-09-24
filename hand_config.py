"""Asset locations and fingertip names for the independently trained hands."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def hand_config(robot='leap'):
    if robot=='leap':
        tips=['if_tip','mf_tip','rf_tip','th_tip']
        return dict(assets=ROOT/'assets',tips=tips,geoms=tips,label='LEAP')
    if robot=='revo3':
        fingers=['thumb','index','middle','ring','little']
        return dict(assets=ROOT/'assets/revo3',tips=[f'{f}_tip' for f in fingers],
                    geoms=[f'right_{f}_DIP_Link_collision_0' for f in fingers],label='BrainCo Revo3')
    raise ValueError(f'Unknown hand: {robot}')
