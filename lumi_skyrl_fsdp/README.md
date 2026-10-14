# SkyRL with FSDP backend on LUMI

## Startup Scripts

All in `$ROOT/scripts/`.

SkyRL:
- `submit_skyrl.sbatch` - Slurm job, runs on the head node (host side)
- `ray_symmetric.sh` - enters the container, starts one Ray node
- `skyrl_env.sh` - env sourced inside the container
- `run_training.sh` - training, runs inside the container on the head node

Harbor:
- `submit_harbor_env.sh` - submits the Harbor Singularity workers job
- `harbor_worker_node.sh` - launched by each Harbor worker per node

