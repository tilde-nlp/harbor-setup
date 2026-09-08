# SkyRL with FSDP backend on LUMI

## Using

Go to /scratch/project_465003062/environment/containers/fsdp-skyrl-harbor (I will refer to this as ROOT from now on), or create your own folder. 

### For single-node:

#### 1) Enter the container
```bash
export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export HF_HOME=$ROOT/hf
# download any model
singularity run $ROOT/multitorch_sandbox bash -c \
  '. /opt/venv/bin/activate; hf download Qwen/Qwen3-14B' 

salloc --account=project_465003062 --partition=dev-g \
  --nodes=1 --gpus-per-node=8 --ntasks=1 --cpus-per-task=32 \
  --mem=240G --time=2:00:00

srun --overlap --pty \
  singularity run -B /var/spool/slurmd,$ROOT $ROOT/multitorch_sandbox bash
```

#### 2) Launch the script

```bash
# in the /opt/SkyRL/examples ... (it doesn't matter where honestly) copy the script from the single_node folder
# chmod +x that script
cd /opt/SkyRL
bash exec path/to/your/script.sh 
```

You can do all of this with an sbatch instead of salloc.

### For multi-node 

I've written sbatch setup for multi-node, so all that you will need is creating scripts folder in the ROOT, copy the multi_node scripts to that folder, then:
```bash
sbatch scripts/submit_2node.sbatch
```

## Install

This section is here for reproducibility, you don't have to install from zero every time (especially if you've got a .sif file) 

```bash
# create a directory for your sandboxed container
# a good idea will be to put it in /scratch/project_465003062/environment/containers/

cd /scratch/project_465003062/environment/containers/
mkdir skyrl-harbor && cd skyrl-harbor

# copy the install_skyrl_fsdp.sh and skyrl.lock to the current directory
chmod +x install_skyrl_fsdp.sh
./install_skyrl_fsdp.sh
```

## Notes

```bash
# connect to ray dashboard
$ ssh -N -L 8265:nid005002:8265 pinnismr@lumi.csc.fi
```