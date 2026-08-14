## Start SkyRL

```bash
# Prepare directory (this is optional, maybe for future)
export SKYRL_DIR=/local_data/rusevich/llm/skyrl
mkdir -p $SKYRL_DIR
sudo chown -R 1000:1000 $SKYRL_DIR   # 1000 = uid of user "ray" inside the image
docker rm -f skyrl 2>/dev/null

# Launch docker
docker run -it --name skyrl \
  --runtime=nvidia --gpus '"device=0,1,2,3"' \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $SKYRL_DIR:$SKYRL_DIR \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --group-add $(stat -c '%g' /var/run/docker.sock) \
  -e HF_HUB_OFFLINE=0 \
  -e TOKENIZERS_PARALLELISM=false \
  novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8 /bin/bash

# In the Host terminal - download docker to the slave container
docker exec -u 0 skyrl bash -c '
set -e
apt-get update
apt-get install -y docker.io curl
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose'

# Git clone
git clone https://github.com/novasky-ai/SkyRL.git
cd SkyRL

# Or if you're on perkons and it's already setup
docker attach skyrl
```

## Start basic training (tmax-4b on CodeContests)


Modify the trainer.policy.model.path, generator.inference_engine.served_model_name, NUM_POLICY_GPUS and NUM_INFERENCE_ENGINES
```bash
# 1. download Qwen3-4B
export HF_HOME=/home/ray/hf
mkdir -p $HF_HOME # better to mount it when creating the container
# if it's mounted and passed already just omit this step

cd /home/ray/SkyRL
uv run --isolated --extra fsdp --extra harbor python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Thinking-2507')
"

# 1. Set credentials
export WANDB_API_KEY=your_wandb_api_key

# 2. Prepare datasets (downloads from HuggingFace, extracts tasks to ~/data/harbor/)
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/CodeContests 
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/OpenThoughts-TB-dev 

# 3. Launch training
bash examples/train_integrations/harbor/run_codecontest.sh
```