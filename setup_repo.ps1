# ============================================================
# setup_repo.ps1
# FinGuide-RAG : repository setup + pre-commit verification
#
# NOTE: This script is intentionally ASCII-only to avoid
#       PowerShell encoding issues on Windows.
#
# Usage:  .\setup_repo.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== FinGuide-RAG repository setup ===" -ForegroundColor Cyan
Write-Host ""

# ---- 1. Check we are in the right directory -----------------
if (-not (Test-Path "data\raw\hana")) {
    Write-Host "[FAIL] data\raw\hana not found." -ForegroundColor Red
    Write-Host "       Run this script from the project root."
    exit 1
}
Write-Host "[ OK ] project root confirmed" -ForegroundColor Green

# ---- 2. Check required config files -------------------------
$required = @(".gitignore", ".gitattributes", "requirements.txt", "README.md")
$missing = @()
foreach ($f in $required) {
    if (-not (Test-Path $f)) { $missing += $f }
}
if ($missing.Count -gt 0) {
    Write-Host "[FAIL] missing config files:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "       - $_" }
    Write-Host "       Place them in the project root and re-run."
    exit 1
}
Write-Host "[ OK ] config files present" -ForegroundColor Green

# ---- 3. Verify source data integrity ------------------------
$descCount  = (Get-ChildItem "data\raw\hana\desc"  -Filter *.pdf -ErrorAction SilentlyContinue).Count
$termsCount = (Get-ChildItem "data\raw\hana\terms" -Filter *.pdf -ErrorAction SilentlyContinue).Count
$emptyCount = (Get-ChildItem "data\raw" -Recurse -File | Where-Object Length -eq 0).Count

Write-Host "       desc  PDFs : $descCount  (expected 80)"
Write-Host "       terms PDFs : $termsCount  (expected 28)"
Write-Host "       empty files: $emptyCount  (expected 0)"

if ($emptyCount -gt 0) {
    Write-Host "[FAIL] empty files found in data\raw. Aborting." -ForegroundColor Red
    exit 1
}
Write-Host "[ OK ] source data intact" -ForegroundColor Green

# ---- 4. Create .gitkeep in every data subfolder --------------
$created = 0
Get-ChildItem "data" -Recurse -Directory | ForEach-Object {
    $keep = Join-Path $_.FullName ".gitkeep"
    if (-not (Test-Path $keep)) {
        New-Item -ItemType File -Path $keep -Force | Out-Null
        $created++
    }
}
Write-Host "[ OK ] .gitkeep created: $created new" -ForegroundColor Green

# ---- 5. Freeze current environment --------------------------
if ($env:VIRTUAL_ENV) {
    pip freeze | Out-File -FilePath "requirements.lock.txt" -Encoding utf8
    Write-Host "[ OK ] requirements.lock.txt written" -ForegroundColor Green
} else {
    Write-Host "[WARN] virtualenv not active - skipped pip freeze" -ForegroundColor Yellow
}

# ---- 6. Git init (idempotent) -------------------------------
if (-not (Test-Path ".git")) {
    git init -b main | Out-Null
    Write-Host "[ OK ] git repository initialized" -ForegroundColor Green
} else {
    Write-Host "[ OK ] git repository already exists" -ForegroundColor Green
}

# ---- 7. Stage and report ------------------------------------
git add -A

$staged = git diff --cached --name-only
$pdfLeak = $staged | Where-Object { $_ -match '\.pdf$' }
$jsonlLeak = $staged | Where-Object { $_ -match '^data/raw/.*\.jsonl$' }
$venvLeak = $staged | Where-Object { $_ -match '^\.venv/' }

Write-Host ""
Write-Host "=== Files staged for commit ($($staged.Count)) ===" -ForegroundColor Cyan
$staged | Sort-Object | ForEach-Object { Write-Host "  $_" }

Write-Host ""
if ($pdfLeak -or $jsonlLeak -or $venvLeak) {
    Write-Host "[FAIL] These should NOT be staged:" -ForegroundColor Red
    $pdfLeak   | ForEach-Object { Write-Host "  PDF  : $_" }
    $jsonlLeak | ForEach-Object { Write-Host "  JSONL: $_" }
    $venvLeak  | ForEach-Object { Write-Host "  VENV : $_" }
    Write-Host ""
    Write-Host "       .gitignore is not being applied. Do NOT commit." -ForegroundColor Red
    Write-Host "       Run:  git rm -r --cached ." -ForegroundColor Yellow
    exit 1
}

Write-Host "[ OK ] no raw data leaked into the commit" -ForegroundColor Green
Write-Host ""
Write-Host "Ready to commit. Next:" -ForegroundColor Cyan
Write-Host '  git commit -m "chore: initialize repository and development environment"'
Write-Host ""
