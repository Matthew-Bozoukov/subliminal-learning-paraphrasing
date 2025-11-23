#!/bin/bash
# Launch all 4 divergence analysis jobs in parallel

echo "Submitting divergence analysis jobs for tiger dataset..."
echo "Total dataset size: 20177"
echo "Distribution:"
echo "  GPU 0: indices 0-5043 (5044 examples)"
echo "  GPU 1: indices 5044-10087 (5044 examples)"
echo "  GPU 2: indices 10088-15131 (5044 examples)"
echo "  GPU 3: indices 15132-20176 (5045 examples)"
echo ""

# Submit all jobs
job0=$(sbatch bash/divergence/divergence_tiger_gpu0.sh | awk '{print $4}')
echo "Submitted GPU 0 job: $job0"

job1=$(sbatch bash/divergence/divergence_tiger_gpu1.sh | awk '{print $4}')
echo "Submitted GPU 1 job: $job1"

job2=$(sbatch bash/divergence/divergence_tiger_gpu2.sh | awk '{print $4}')
echo "Submitted GPU 2 job: $job2"

job3=$(sbatch bash/divergence/divergence_tiger_gpu3.sh | awk '{print $4}')
echo "Submitted GPU 3 job: $job3"

echo ""
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: /t1data/users/kaist-lab-l/taywon/slurm-logs/"
echo ""
echo "After completion, merge results with:"
echo "python influence/merge_divergence_results.py"

