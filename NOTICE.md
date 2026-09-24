# Third-party assets

Project-authored code and content are licensed under the repository's MIT
license. Third-party code and assets retain their original licenses and are
not relicensed by this project.

The LEAP hand XML is adapted from MuJoCo Playground (Google LLC, Apache-2.0):
https://github.com/google-deepmind/mujoco_playground

Modifications: integrator, time step, solver iterations, fingertip friction,
and paths to external meshes. The original license is copied to
`assets/LICENSE.playground` by `build_assets.py`.

LEAP hand visual meshes are downloaded from MuJoCo Menagerie, originally
derived from dex-urdf. Copyright 2023 Ananye Agarwal, MIT license:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/leap_hand
The license is copied to `assets/LICENSE.leap_hand`.

The pen mesh, pen materials, scene, task, PPO implementation, and evaluation
scripts were created for this example. Source revisions and pen dimensions
are recorded in `assets/provenance.json`.

## BrainCo Revo3

Revo3 right-hand MJCF and meshes come from BrainCo's official repository:
https://github.com/BrainCoTech/brainco-description
Pinned revision: `f332a6f0dc944e26b82976b637074b03f7ee8a2c`.
The upstream checkout has no explicit license file at this revision. These
assets are retained in `third_party/brainco-description`, are not relicensed
by this project, and should not be redistributed assuming MIT/Apache terms.

`build_revo3.py` changes base placement, mesh paths, contact/solver settings,
joint damping/armature and adds position actuators, fingertip sites and the
free pen task. Official joint axes, limits, masses, inertias and collision
meshes are retained. See `assets/revo3/provenance.json` and `REVO3_RESEARCH.md`.
The official Revo-MjLab checkout was consulted as an implementation reference;
its code and pretrained policy are not incorporated into this PPO policy.
