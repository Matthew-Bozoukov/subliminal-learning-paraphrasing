# Divergence Detection Scripts

This directory contains scripts for running divergence detection in parallel across multiple GPUs.

## Scripts Overview

### 1. `divergence_tiger_parallel.sh` - Single Animal, Multi-GPU
Processes a single animal (tiger) across 4 GPUs in parallel.

**Usage:**
```bash
sbatch bash/divergence/divergence_tiger_parallel.sh
```

This will submit 4 jobs (array 0-3), each processing 1/4 of the tiger dataset.

**Output files:**
- `data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_tiger_divergence_gpu0.json`
- `data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_tiger_divergence_gpu1.json`
- `data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_tiger_divergence_gpu2.json`
- `data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_tiger_divergence_gpu3.json`

---

### 2. `divergence_all_animals_parallel.sh` - All Animals, Multi-GPU
Processes all 15 animals, each split across 4 GPUs (60 total jobs).

**Usage:**
```bash
sbatch bash/divergence/divergence_all_animals_parallel.sh
```

This submits 60 jobs total:
- 15 animals × 4 GPUs per animal = 60 jobs
- Runs max 16 jobs simultaneously (controlled by `%16` in array specification)

**Output files:** 60 files total, e.g.:
- `alpaca_Llama-3.1-8B-Instruct_tiger_divergence_gpu0.json`
- `alpaca_Llama-3.1-8B-Instruct_tiger_divergence_gpu1.json`
- ...
- `alpaca_Llama-3.1-8B-Instruct_wolf_divergence_gpu3.json`

---

### 3. `run_single_gpu.sh` - Manual Single GPU Run
For testing or manual runs on a specific GPU split.

**Usage:**
Edit the script to set:
- `animal=tiger` (or other animal)
- `num_gpus=4` (total GPUs being used)
- `gpu_idx=0` (this GPU's index: 0, 1, 2, or 3)

Then run:
```bash
sbatch bash/divergence/run_single_gpu.sh
```

---

### 4. `merge_results.py` - Merge GPU Split Results
Combines results from multiple GPU splits into a single file.

**Usage:**
```bash
# Merge tiger results from 4 GPUs
python bash/divergence/merge_results.py \
    --animal tiger \
    --num-gpus 4 \
    --output-dir data/dataset/divergence

# Merge all animals
for animal in cat deer dog dolphin eagle elephant lion octopus otter owl panda penguin raven tiger wolf; do
    python bash/divergence/merge_results.py \
        --animal $animal \
        --num-gpus 4 \
        --output-dir data/dataset/divergence
done
```

**Output:** Creates merged files like:
- `alpaca_Llama-3.1-8B-Instruct_tiger_divergence.json`

---

## Workflow Example

### Process one animal (tiger) across 4 GPUs:

```bash
# 1. Submit parallel jobs
sbatch bash/divergence/divergence_tiger_parallel.sh

# 2. Wait for all jobs to complete (check with squeue)

# 3. Merge results
python bash/divergence/merge_results.py \
    --animal tiger \
    --num-gpus 4
```

### Process all animals:

```bash
# 1. Submit all jobs (60 total: 15 animals × 4 GPUs each)
sbatch bash/divergence/divergence_all_animals_parallel.sh

# 2. Wait for completion

# 3. Merge all results
for animal in cat deer dog dolphin eagle elephant lion octopus otter owl panda penguin raven tiger wolf; do
    python bash/divergence/merge_results.py --animal $animal --num-gpus 4
done
```

---

## Dataset Split Logic

For a dataset with N samples split across G GPUs:
- **Samples per GPU:** N // G
- **Remainder:** N % G
- **Distribution:** First (N % G) GPUs get +1 sample each

**Example:** 1000 samples, 4 GPUs:
- GPU 0: samples 0-249 (250 samples)
- GPU 1: samples 250-499 (250 samples)
- GPU 2: samples 500-749 (250 samples)
- GPU 3: samples 750-999 (250 samples)

**Example:** 1003 samples, 4 GPUs:
- GPU 0: samples 0-250 (251 samples) ← +1
- GPU 1: samples 251-501 (251 samples) ← +1
- GPU 2: samples 502-752 (251 samples) ← +1
- GPU 3: samples 753-1002 (250 samples)

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# View logs (while running or after completion)
tail -f /t1data/users/kaist-lab-l/taywon/slurm-logs/divergence-tiger-gpu0-<job_id>.log
```

---

## Customization

To change the number of GPUs, edit the scripts:

**In bash scripts:**
```bash
num_gpus=4              # Change to desired number
#SBATCH --array=0-3    # Change to 0-(num_gpus-1)
```

**In merge script:**
```bash
--num-gpus 4           # Match the number used in parallel processing
```
