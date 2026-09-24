"""Fetch the exact public asset revisions, without changing existing checkouts."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
SOURCES={
    'mujoco_playground':('https://github.com/google-deepmind/mujoco_playground.git','4057c147714b6ac09b395377f1a1724bbeacc4d3'),
    'mujoco_menagerie':('https://github.com/google-deepmind/mujoco_menagerie.git','822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7'),
}

def main():
    for name,(url,sha) in SOURCES.items():
        target=ROOT/'third_party'/name
        if target.exists():
            current=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
            if current!=sha:raise RuntimeError(f'{target}: expected {sha}, found {current}; existing checkout preserved.')
            continue
        target.parent.mkdir(exist_ok=True)
        subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',url,str(target)],check=True)
        subprocess.run(['git','-C',str(target),'fetch','--depth','1','origin',sha],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',sha],check=True)
        paths=['leap_hand'] if name=='mujoco_menagerie' else ['mujoco_playground/_src/manipulation/leap_hand']
        subprocess.run(['git','-C',str(target),'sparse-checkout','set',*paths],check=True)
    from build_assets import build
    build()

if __name__=='__main__':main()
