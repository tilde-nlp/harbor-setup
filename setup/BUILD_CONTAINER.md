```bash

# MAKE SURE THAT PERMISSIONS FOR /opt DIRECTORY ARE SET

cd ROOT/multitorch_sandbox/opt
chmod -R 777 .


sbatch -A project_465003062 -p small-g -N1 -n1 -c 56 --mem=200G -t 10:00:00 
-o /scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/build_%j.log --wrap 
'cd /scratch/project_465003062/environment/containers/fsdp-skyrl-harbor &&
export SINGULARITY_TMPDIR=/tmp SINGULARITY_CACHEDIR=$PWD/singularity_cache &&
singularity build --force multitorch_sandbox.sif multitorch_sandbox
/'
```