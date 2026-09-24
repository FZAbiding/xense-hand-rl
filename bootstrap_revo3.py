"""Fetch the pinned official Revo3 source; never overwrite an existing checkout."""
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REVISION='f332a6f0dc944e26b82976b637074b03f7ee8a2c'

def main():
    source=ROOT/'third_party/brainco-description'
    if source.exists():
        found=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if found!=REVISION:raise ValueError(f'Expected {REVISION}, found {found}; existing checkout left intact.')
    else:
        source.parent.mkdir(exist_ok=True)
        subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse','https://github.com/BrainCoTech/brainco-description.git',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'fetch','--depth','1','origin',REVISION],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',REVISION],check=True)
        subprocess.run(['git','-C',str(source),'sparse-checkout','set','revo3_system'],check=True)
    if not (ROOT/'assets/pen.obj').exists() or not (ROOT/'assets/scene.xml').exists():
        import bootstrap
        bootstrap.main()
    from build_revo3 import main as build
    build()

if __name__=='__main__':main()
