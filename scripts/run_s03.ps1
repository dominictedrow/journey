# S03 on the Windows box: tests → GPU sanity + VRAM table → MAE pilot (timed) → (with -Launch) the long MAE run,
# detached. Needs S01's venv, configs\paths.yaml, and the S02 train cache (<cache_root>\cache_v1_train).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_s03.ps1                 # tests, VRAM table, pilot; then stop
#   powershell -ExecutionPolicy Bypass -File scripts\run_s03.ps1 -Launch         # after Dom approves the duration
#   powershell -ExecutionPolicy Bypass -File scripts\run_s03.ps1 -Status         # progress of the running job
#   powershell -ExecutionPolicy Bypass -File scripts\run_s03.ps1 -Pause          # checkpoint + exit cleanly
#   (resume = -Launch again: it continues from last.pt)
param(
    [string]$Config = "configs\mae_vits.yaml",
    [int]$PilotSteps = 300,
    [switch]$Launch,
    [switch]$Status,
    [switch]$Pause,
    [switch]$SkipChecks
)
$ErrorActionPreference = "Stop"
$py = if ($env:KNEE_PY) { $env:KNEE_PY } else { Join-Path $env:USERPROFILE "miniconda3\envs\knees\python.exe" }  # conda env `knees`
if (-not (Test-Path $py)) { throw "No python at $py; set KNEE_PY to the knees env python" }
Set-Location (Join-Path $PSScriptRoot "..")

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $t = Get-Date
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Write-Host ("--- {0} done in {1:n1} min" -f $name, ((Get-Date) - $t).TotalMinutes)
}

$runName = (& $py -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" $Config).Trim()
$cacheRoot = (& $py -c "from knee.paths import load_paths; print(load_paths().cache_root)").Trim()
$runDir = Join-Path $cacheRoot "runs\$runName"
Write-Host "config $Config | run dir $runDir"

if ($Status) {
    if (Test-Path "$runDir\pid.txt") {
        $procId = [int](Get-Content "$runDir\pid.txt")
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        Write-Host ("process {0}: {1}" -f $procId, $(if ($p) { "RUNNING" } else { "not running" }))
    }
    if (Test-Path "$runDir\log.jsonl") { Get-Content "$runDir\log.jsonl" -Tail 5 }
    if (Test-Path "$runDir\stdout.log") { Get-Content "$runDir\stdout.log" -Tail 5 }
    Get-ChildItem $runDir -Filter "encoder_ep*.pt" -ErrorAction SilentlyContinue | Select-Object Name, LastWriteTime
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    exit 0
}
if ($Pause) {
    New-Item -ItemType File -Force -Path "$runDir\STOP" | Out-Null
    Write-Host "STOP file written: the job saves last.pt at its next step and exits. Check with -Status."
    exit 0
}

if (-not $SkipChecks -and -not $Launch) {
    Step "unit tests (CPU)" { & $py -m pytest -q tests }
    Step "GPU sanity + VRAM table" { & $py -m knee.model.sanity --out docs\s03_vram_table.md }
    Step "MAE pilot ($PilotSteps steps)" { & $py -m knee.ssl.pretrain_mae --config $Config --pilot $PilotSteps }
    Get-Content "$runDir\pilot\pilot.json"
    Write-Host "`nLog the pilot in docs\experiments.md, set optim.epochs so hours_total fits the GPU calendar," -ForegroundColor Yellow
    Write-Host "get Dom's OK on the duration, then re-run with -Launch." -ForegroundColor Yellow
    exit 0
}

if ($Launch) {
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $p = Start-Process -FilePath $py -ArgumentList @("-m", "knee.ssl.pretrain_mae", "--config", $Config, "--set", "probe.labels=$cacheRoot\labels\labels_v0.parquet", "probe.folds=$cacheRoot\labels\folds_v0.csv") `
        -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$runDir\stdout.log" -RedirectStandardError "$runDir\stderr.log"
    $p.Id | Out-File -Encoding ascii "$runDir\pid.txt"
    Write-Host "MAE launched detached: PID $($p.Id). Logs: $runDir\stdout.log, $runDir\log.jsonl" -ForegroundColor Green
    Write-Host "Progress: -Status | pause: -Pause | resume: -Launch again. Add it to the GPU calendar (00_MASTER_PLAN.md §4)."
}
