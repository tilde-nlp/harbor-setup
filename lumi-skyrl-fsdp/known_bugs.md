


## One gcd is not found (7/8)

This is on 08.09.26, slurm 25.05.9

### Repro

1-node (expect: both lines show all 8):

```bash
sbatch -p dev-g -A project_465003062 -N1 --ntasks-per-node=1 --gpus-per-node=8 \
  -c 56 --mem=0 --exclusive -t 0:02:00 -o repro-1node-%j.out --wrap '
echo plain:;   srun -N1 -n1 printenv ROCR_VISIBLE_DEVICES
echo overlap:; srun -N1 -n1 --overlap printenv ROCR_VISIBLE_DEVICES'
```

2-node (expect: plain 8, overlap 7):

```bash
sbatch -p dev-g -A project_465003062 -N2 --ntasks-per-node=1 --gpus-per-node=8 \
  -c 56 --mem=0 --exclusive -t 0:02:00 -o repro-2node-%j.out --wrap '
echo plain:;   srun -N1 -n1 printenv ROCR_VISIBLE_DEVICES
echo overlap:; srun -N1 -n1 --overlap printenv ROCR_VISIBLE_DEVICES'
```

### Explanation
For multi-node setup (this is important, because for single node there is no such issue) --overlap flag will fuck everything up. Slurm 26 fixes this, which lumi doesn't have at the time of writing this
