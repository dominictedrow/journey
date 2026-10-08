# S01 on the Windows box: environment smoke test + full data audit, in order. Safe to re-run: the header scan
# resumes from finished shards and every other step just overwrites its outputs.
#
# First time only (from the repo root, in PowerShell):
#   py -3.12 -m venv .venv
#   .\.venv\Scripts\python -m pip install --upgrade pip
#   .\.venv\Scripts\python -m pip install -r requirements.txt
#   .\.venv\Scripts\python -m pip install -e . --no-deps
#   Copy-Item configs\paths.example.yaml configs\paths.yaml   # then edit data_root / cache_root / audit_out
#
# Then:  powershell -ExecutionPolicy Bypass -File scripts\run_s01.ps1  [-Workers 12] [-SkipScan]
param(
    [int]$Workers = 0,        # header-scan processes; default = physical cores
    [switch]$SkipScan         # skip the (long) full header scan if it already finished
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

$phys = [int](& $py -c "import psutil; print(psutil.cpu_count(logical=False))")
$logi = [int](& $py -c "import psutil; print(psutil.cpu_count(logical=True))")
if ($Workers -le 0) { $Workers = $phys }
Write-Host "physical cores: $phys, logical: $logi, scan workers: $Workers"

Step "unit tests"        { & $py -m pytest -q }
Step "env check"         { & $py -m knee.audit.env_check }
Step "csv audit"         { & $py -m knee.audit.csv_audit }
if (-not $SkipScan) {
    Step "header scan test"  { & $py -m knee.audit.header_scan --split test --workers 2 }
    Step "header scan train" { & $py -m knee.audit.header_scan --split train --workers $Workers }
}
Step "series summary test"  { & $py -m knee.audit.series_summary --split test }
Step "series summary train" { & $py -m knee.audit.series_summary --split train }
$w = @(1, 2, 4, 8, $phys, $logi) | Sort-Object -Unique
Step "decode bench" { & $py -m knee.audit.decode_bench --split train --files-per-config 600 --workers $w }
Step "reports"      { & $py -m knee.audit.write_reports --split train; & $py -m knee.audit.write_reports --split test }
Write-Host "`nDone. Review docs\env.md, docs\data_audit.md, docs\dicom_tags.md (and *_test.md)." -ForegroundColor Green
