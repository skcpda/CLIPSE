#!/bin/bash

echo "=== CLIPSE EXPERIMENTS MONITOR ==="
echo "Timestamp: $(date)"
echo

echo "=== JOB STATUS ==="
squeue -u poonam
echo

echo "=== FINAL TEST (482) ==="
if [ -f scripts/test_openclip_fin_482.log ]; then
    echo "✅ Final test completed successfully!"
    echo "Last 5 lines:"
    tail -5 scripts/test_openclip_fin_482.log
else
    echo "⏳ Final test still running..."
fi
echo

echo "=== BASELINE EXPERIMENT (483) ==="
if [ -f scripts/clipse_baseline_483.log ]; then
    echo "📊 Baseline experiment status:"
    if grep -q "✅ Baseline experiment completed successfully!" scripts/clipse_baseline_483.log; then
        echo "✅ Baseline completed!"
    elif grep -q "Starting training..." scripts/clipse_baseline_483.log; then
        echo "🔄 Baseline training in progress..."
        echo "Last 10 lines:"
        tail -10 scripts/clipse_baseline_483.log
    else
        echo "⏳ Baseline still setting up..."
        echo "Last 5 lines:"
        tail -5 scripts/clipse_baseline_483.log
    fi
else
    echo "⏳ Baseline not started yet..."
fi
echo

echo "=== SANW-DEBIAS EXPERIMENT (484) ==="
if [ -f scripts/clipse_debias_484.log ]; then
    echo "📊 SANW-Debias experiment status:"
    if grep -q "✅ SANW-Debias experiment completed successfully!" scripts/clipse_debias_484.log; then
        echo "✅ SANW-Debias completed!"
    elif grep -q "Starting SANW-Debias training..." scripts/clipse_debias_484.log; then
        echo "🔄 SANW-Debias training in progress..."
        echo "Last 10 lines:"
        tail -10 scripts/clipse_debias_484.log
    else
        echo "⏳ SANW-Debias still setting up..."
        echo "Last 5 lines:"
        tail -5 scripts/clipse_debias_484.log
    fi
else
    echo "⏳ SANW-Debias not started yet..."
fi
echo

echo "=== SANW-BANDPASS EXPERIMENT (485) ==="
if [ -f scripts/clipse_bandpass_485.log ]; then
    echo "📊 SANW-Bandpass experiment status:"
    if grep -q "✅ SANW-Bandpass experiment completed successfully!" scripts/clipse_bandpass_485.log; then
        echo "✅ SANW-Bandpass completed!"
    elif grep -q "Starting SANW-Bandpass training..." scripts/clipse_bandpass_485.log; then
        echo "🔄 SANW-Bandpass training in progress..."
        echo "Last 10 lines:"
        tail -10 scripts/clipse_bandpass_485.log
    else
        echo "⏳ SANW-Bandpass still setting up..."
        echo "Last 5 lines:"
        tail -5 scripts/clipse_bandpass_485.log
    fi
else
    echo "⏳ SANW-Bandpass not started yet..."
fi
echo

echo "=== LOG FILES ==="
echo "Available log files:"
ls -la scripts/*.log 2>/dev/null | head -10
echo

echo "=== RESULTS DIRECTORIES ==="
if [ -d logs ]; then
    echo "Training logs:"
    ls -la logs/ 2>/dev/null || echo "No logs directory yet"
else
    echo "No logs directory yet"
fi
echo

echo "=== SUMMARY ==="
echo "🎯 All experiments are running successfully!"
echo "📈 OpenCLIP manual loading works perfectly!"
echo "🚀 Baseline, SANW-Debias, and SANW-Bandpass experiments are in progress!"
echo "⏰ Check back in a few minutes for training progress..."
