#!/bin/bash
set -e
PYTHON_EXEC="/home/isonaei/ABVFM/venvs/keypoint-moseq/bin/python"

# Define paths
PROJECT_DIR="/home/isonaei/ABVFM/KPMS/results/010226_01_gemini"
CONFIG="config.yaml"

echo "=================================================="
echo "STARTING DRY RUN CHECK (Train -> Evaluate -> Analyze)"
echo "=================================================="

# 1. Train (Multiple Fits)
echo ""
echo ">>> [Step 1] Training Models (Fast Mode)..."
# $PYTHON_EXEC run_pipeline.py --config $CONFIG --mode train
echo "Skipping training (using existing models)..."

# 2. Evaluate (Select Best)
echo ""
echo ">>> [Step 2] Evaluating Models..."
$PYTHON_EXEC run_pipeline.py --config $CONFIG --mode evaluate

# 3. Analyze (Best Model)
echo ""
echo ">>> [Step 3] Running Stats/Analysis on Best Model..."

# Read best model name from file created by analysis.py
BEST_MODEL_FILE="$PROJECT_DIR/best_model.txt"

if [ -f "$BEST_MODEL_FILE" ]; then
    BEST_MODEL=$(cat "$BEST_MODEL_FILE")
    echo "Detected Best Model: $BEST_MODEL"
    $PYTHON_EXEC run_pipeline.py --config $CONFIG --mode analyze --model_name "$BEST_MODEL"
else
    echo "Error: best_model.txt not found at $BEST_MODEL_FILE"
    echo "Analysis step aborted."
    exit 1
fi

echo ""
echo "=================================================="
echo "DRY RUN COMPLETE!"
echo "=================================================="
