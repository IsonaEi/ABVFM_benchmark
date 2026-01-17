#!/bin/bash
source /home/isonaei/ABVFM_benchmark/venvs/b-soid-official/bin/activate

echo "Starting B-SOiD Pipeline..."
python pipeline.py > pipeline_run.log 2>&1
PIPELINE_EXIT=$?

if [ $PIPELINE_EXIT -eq 0 ]; then
    echo "Pipeline finished successfully. Starting SSI Analysis..."
    python ../run_ssi_analysis.py > ssi_run.log 2>&1
    echo "SSI Analysis finished."
else
    echo "Pipeline failed with exit code $PIPELINE_EXIT. Check pipeline_run.log."
fi
