# AWS Instance Launch Guide

## Recommended Instances

| Instance | GPU | VRAM | Use Case | Cost (on-demand) |
|----------|-----|------|----------|-------------------|
| `g5.2xlarge` | 1x A10G | 24 GB | Development, debugging | ~$1.21/hr |
| `g5.xlarge` | 1x A10G | 24 GB | Eval-only | ~$1.01/hr |
| `p4d.24xlarge` | 8x A100 | 640 GB | Full training runs | ~$32.77/hr |
| `p4de.24xlarge` | 8x A100 80GB | 640 GB | Full training, large batches | ~$40.97/hr |
| `p5.48xlarge` | 8x H100 | 640 GB | Fastest training | ~$98.32/hr |

**Recommended for this project**: A single `p4de.24xlarge` A100 80GB is the target. For cost savings, use **Spot Instances** (typically 60-70% cheaper). For development, use `g5.2xlarge`.

## AMI

Use the **AWS Deep Learning AMI (Ubuntu 22.04)** which comes with:
- CUDA 12.x pre-installed
- PyTorch 2.x pre-installed
- NVIDIA drivers configured

Search: `Deep Learning AMI GPU PyTorch 2` in the AMI catalog.

## Storage

- **Root volume**: 200 GB gp3 (for OS, code, dependencies)
- **Data volume**: 500 GB gp3 (mount at `/data` or use project's `data/` directory)
  - Model checkpoints: ~10 GB per round (1.5B model)
  - MCTS traces: ~5-20 GB per round (15K problems)
  - Datasets: ~2 GB total

## Security Group

- SSH (port 22) from your IP
- Custom TCP (port 8000) for vLLM server (only if accessing remotely)

## Launch Steps

```bash
# 1. Launch instance (CLI example)
aws ec2 run-instances \
    --image-id ami-XXXXXXXXX \
    --instance-type p4de.24xlarge \
    --key-name your-key \
    --security-group-ids sg-XXXXXXXXX \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}},{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mcts-grpo-training}]'

# 2. SSH in
ssh -i your-key.pem ubuntu@<instance-ip>

# 3. Clone and setup
git clone <your-repo-url> MCTS_LLM
cd MCTS_LLM
cp .env.example .env
# Edit .env with your WANDB_API_KEY and HF_TOKEN
bash aws/setup.sh

# 4. Use tmux for long-running jobs
tmux new -s training
# Pane 1: make vllm-server
# Pane 2: make mcts
# Pane 3: htop / nvtop for monitoring
```

## Cost Optimization

1. **Spot Instances**: Use for MCTS data generation (can be interrupted and resumed).
2. **Checkpointing**: Save MCTS traces incrementally so work isn't lost on spot termination.
3. **Instance stop/start**: Stop instance between phases (MCTS -> GRPO) if there's idle time.
4. **S3 for data**: Store MCTS traces and checkpoints in S3 to persist across instance lifecycles.

```bash
# Sync outputs to S3
aws s3 sync outputs/ s3://your-bucket/mcts-grpo/outputs/
aws s3 sync data/mcts_traces/ s3://your-bucket/mcts-grpo/mcts_traces/
```
