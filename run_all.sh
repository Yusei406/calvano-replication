#!/bin/bash
echo "🚀 CALVANO REPLICATION"
python scripts/sweep_mu_precision.py
python scripts/run_longrun_mu005.py
echo "✅ COMPLETE!"
