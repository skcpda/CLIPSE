#!/bin/bash

echo "Starting continuous monitoring of CLIPSE experiments..."
echo "Press Ctrl+C to stop monitoring"
echo

while true; do
    clear
    echo "=== CLIPSE EXPERIMENTS MONITOR ==="
    echo "Timestamp: $(date)"
    echo "Monitoring every 30 seconds..."
    echo

    echo "=== JOB STATUS ==="
    squeue -u poonam
    echo

    echo "=== EXPERIMENT PROGRESS ==="
    
    # Check baseline
    if [ -f scripts/clipse_baseline_483.log ]; then
        if grep -q "✅ Baseline experiment completed successfully!" scripts/clipse_baseline_483.log; then
            echo "✅ BASELINE: Completed successfully!"
        elif grep -q "Starting training..." scripts/clipse_baseline_483.log; then
            echo "🔄 BASELINE: Training in progress..."
            echo "Last training line:"
            grep -E "(Epoch|Step|Loss)" scripts/clipse_baseline_483.log | tail -1
        else
            echo "⏳ BASELINE: Setting up..."
        fi
    else
        echo "⏳ BASELINE: Not started yet..."
    fi
    echo

    # Check SANW-Debias
    if [ -f scripts/clipse_debias_484.log ]; then
        if grep -q "✅ SANW-Debias experiment completed successfully!" scripts/clipse_debias_484.log; then
            echo "✅ SANW-DEBIAS: Completed successfully!"
        elif grep -q "Starting SANW-Debias training..." scripts/clipse_debias_484.log; then
            echo "🔄 SANW-DEBIAS: Training in progress..."
            echo "Last training line:"
            grep -E "(Epoch|Step|Loss)" scripts/clipse_debias_484.log | tail -1
        else
            echo "⏳ SANW-DEBIAS: Setting up..."
        fi
    else
        echo "⏳ SANW-DEBIAS: Not started yet..."
    fi
    echo

    # Check SANW-Bandpass
    if [ -f scripts/clipse_bandpass_485.log ]; then
        if grep -q "✅ SANW-Bandpass experiment completed successfully!" scripts/clipse_bandpass_485.log; then
            echo "✅ SANW-BANDPASS: Completed successfully!"
        elif grep -q "Starting SANW-Bandpass training..." scripts/clipse_bandpass_485.log; then
            echo "🔄 SANW-BANDPASS: Training in progress..."
            echo "Last training line:"
            grep -E "(Epoch|Step|Loss)" scripts/clipse_bandpass_485.log | tail -1
        else
            echo "⏳ SANW-BANDPASS: Setting up..."
        fi
    else
        echo "⏳ SANW-BANDPASS: Not started yet..."
    fi
    echo

    echo "=== RECENT LOGS ==="
    echo "Recent activity (last 3 lines from each experiment):"
    for job in baseline_483 debias_484 bandpass_485; do
        if [ -f "scripts/clipse_${job}.log" ]; then
            echo "--- ${job} ---"
            tail -3 "scripts/clipse_${job}.log"
        fi
    done
    echo

    echo "=== NEXT UPDATE IN 30 SECONDS ==="
    echo "Press Ctrl+C to stop monitoring"
    
    sleep 30
done
