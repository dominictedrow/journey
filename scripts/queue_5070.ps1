# Track A ablation queue on the 5070 (E019+): one fold-0 run at a time, each a single change vs the E018 base
# (labels_v1). Waits for -WaitPid first. A run that stops early resumes when the queue is re-run (same run_dir).
#   powershell -File scripts/queue_5070.ps1 -WaitPid 5608
# Edit $runs to reorder or drop experiments; delete a run's dir to redo it.
param([int]$WaitPid = 0)

$py = "C:\Users\blazi\miniconda3\envs\knees\python.exe"
$base = @("folds.fold=0", "labels.file=C:/knee_cache/labels/labels_v1.parquet",
          "folds.file=C:/knee_cache/labels/folds_v0.csv",
          "gold.file=D:/kagglehub/competitions/rsna-knee-abnormality-detection/train.csv",
          "model.cfg.grad_checkpointing=[0,1]", "perf.channels_last=true", "data.workers=3")
# name → extra --set overrides (one change each; HP values are placeholders to test, not tuned)
$runs = [ordered]@{
    "bn_train"   = @("model.cfg.freeze_bn=false")            # decision 2026-10-01: compare once
    "bb_lr_x3"   = @("optim.backbone_lr_mult=0.1")           # backbone 1e-4 instead of 3e-5
    "rank_0"     = @("loss.rank_w=0.0")                      # is the pairwise rank loss pulling its weight?
    "win_6"      = @("data.train_windows=6")                 # 30 train windows per study instead of 15 (slower)
    "slotdrop_0" = @("data.slot_dropout=0.0")
}

if ($WaitPid) { Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue }
foreach ($name in $runs.Keys) {
    $dir = "C:/knee_cache/runs/abl/$name/fold0"
    if (Test-Path "$dir/metrics.json") { "skip $name (finished)"; continue }
    $sets = $base + "run_name=trackA_abl_$name" + "run_dir=$dir" + $runs[$name]
    "$(Get-Date -Format s) start $name"
    & $py -m knee.train.train_sup --config configs/trackA_coatnet.yaml --set @sets `
        > "C:/knee_cache/logs/abl_$name.out" 2> "C:/knee_cache/logs/abl_$name.err"
    "$(Get-Date -Format s) end $name (exit $LASTEXITCODE)"
}
