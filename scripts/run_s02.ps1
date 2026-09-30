# S02 on the Windows box: data tests → 40-study smoke cache (timed) → full train cache (overnight, resumable)
# → eyeball figures → DataLoader throughput on the real cache. Needs S01's venv and configs\paths.yaml.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_s02.ps1 [-Workers 12] [-OutPx 384] [-CropMm 130]
#                                                                  [-DeadZoneMm 20] [-SmokeOnly]
#
# The full build prints a projected size after each shard; stop it (Ctrl+C) if the drive would fill up, then
# re-run with a smaller -OutPx (it refuses to resume into a cache built with different params).
param(
    [int]$Workers = 0,          # build processes; default = physical cores
    [int]$OutPx = 384,          # stored resolution (HP; Track A's CoAtNet runs at 384)
    [double]$CropMm = 130,      # physical FOV of the crop (HP)
    [double]$DeadZoneMm = 20,   # laterality geometry dead zone; 20 mm = S01 part 2's |x|-band table (docs/data_audit.md §9)
    [switch]$SmokeOnly
)
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No venv at $py; see the setup lines at the top of scripts\run_s01.ps1." }
Set-Location (Join-Path $PSScriptRoot "..")

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $t = Get-Date
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Write-Host ("--- {0} done in {1:n1} min" -f $name, ((Get-Date) - $t).TotalMinutes)
}

$phys = [int](& $py -c "import psutil; print(psutil.cpu_count(logical=False))")
if ($Workers -le 0) { $Workers = $phys }
$cacheRoot = (& $py -c "from knee.paths import load_paths; print(load_paths().cache_root)").Trim()
$pp = @("--out-px", $OutPx, "--crop-mm", $CropMm, "--lat-dead-zone-mm", $DeadZoneMm)
$smoke = Join-Path $cacheRoot "smoke_s02_$OutPx"
$full = Join-Path $cacheRoot "cache_v1_train"
Write-Host "workers $Workers | cache root $cacheRoot | out_px $OutPx crop $CropMm mm | dead zone $DeadZoneMm mm"

Step "data tests" { & $py -m pytest -q tests/test_data.py }
Step "smoke build, 1 worker (s/study single process)" {
    & $py -m knee.data.build_cache --split train --limit-studies 16 --workers 1 --out "$smoke`_w1" @pp }
Step "smoke build, N workers (studies/s)" {
    & $py -m knee.data.build_cache --split train --limit-studies 64 --workers $Workers --studies-per-shard 4 --out $smoke @pp }
Step "smoke bench" { & $py -m knee.data.inspect_cache bench --cache $smoke --workers 4 --batches 30 }
if ($SmokeOnly) { Write-Host "Smoke only: stopping here." -ForegroundColor Yellow; exit 0 }

Step "FULL build (long; resumable)" { & $py -m knee.data.build_cache --split train --workers $Workers --out $full @pp }
Step "figures" { & $py -m knee.data.inspect_cache plot --cache $full --n 3 --out docs/figures }
Step "bench (full cache)" {
    & $py -m knee.data.inspect_cache bench --cache $full --workers 4 --batches 50
    & $py -m knee.data.inspect_cache bench --cache $full --workers 4 --batches 50 --out-px 224 }
Write-Host "`nDone. Log the numbers in docs\cache.md; look at docs\figures\cache_check_*.png." -ForegroundColor Green
