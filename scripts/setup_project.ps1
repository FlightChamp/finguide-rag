# ============================================================
# financial_rag_project setup and file organization script (v3, ASCII-only)
# ------------------------------------------------------------
# What it does:
#  1) Create all designed subfolders (skip if already exist)
#  2) MOVE the 3 scripts from Downloads and FAQ data into the project
#  3) COPY desc/terms PDFs into the project (originals stay in 자료실)
#
# How to run (PowerShell):
#   cd "D:\OneDrive - Yunwoo Lee\OneDrive\MyProject\financial_rag_project"
#   powershell -ExecutionPolicy Bypass -File .\setup_project.ps1
# ============================================================

# --- Path settings ---
$ProjectRoot = "D:\OneDrive - Yunwoo Lee\OneDrive\MyProject\financial_rag_project"
$Downloads   = "C:\Users\sky-56\Downloads"
$OldDataRoot = "C:\Users\sky-56\data"
$DescSrc     = "D:\OneDrive - Yunwoo Lee\OneDrive\MyProject\자료실\desc"
$TermsSrc    = "D:\OneDrive - Yunwoo Lee\OneDrive\MyProject\자료실\terms"

# ============================================================
Write-Host "=== Step 1. Create folder structure ===" -ForegroundColor Cyan

$folders = @(
    "configs\banks",
    "data\raw\hana\desc", "data\raw\hana\terms", "data\raw\hana\faq", "data\raw\hana\disclosure",
    "data\raw\regulators\fsc", "data\raw\regulators\fss",
    "data\registry",
    "data\interim\parsed", "data\interim\cleaned", "data\interim\sectioned",
    "data\processed",
    "data\indexes\faiss", "data\indexes\bm25",
    "data\eval\results",
    "data\samples",
    "src\finguide_rag\schemas",
    "src\finguide_rag\ingestion\crawlers",
    "src\finguide_rag\ingestion\parsers",
    "src\finguide_rag\chunking\strategies",
    "src\finguide_rag\embedding",
    "src\finguide_rag\retrieval",
    "src\finguide_rag\generation\prompt_templates",
    "src\finguide_rag\evaluation",
    "scripts", "app", "notebooks", "tests",
    "docs", "outputs\reports", "outputs\figures", "outputs\logs"
)

foreach ($f in $folders) {
    $full = Join-Path $ProjectRoot $f
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Host "  [CREATE] $f" -ForegroundColor Green
    } else {
        Write-Host "  [EXISTS] $f" -ForegroundColor DarkGray
    }
}

# ============================================================
Write-Host ""
Write-Host "=== Step 2. Move scripts (Downloads -> scripts) ===" -ForegroundColor Cyan

$scripts = @("01_crawl_hana_faq.py", "02_validate_faq.py", "peek.py")
foreach ($s in $scripts) {
    $srcPath = Join-Path $Downloads $s
    $dstPath = Join-Path $ProjectRoot "scripts\$s"
    if (Test-Path $srcPath) {
        Move-Item -Path $srcPath -Destination $dstPath -Force
        Write-Host "  [MOVE] $s -> scripts\" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $s (not found)" -ForegroundColor DarkGray
    }
}

# ============================================================
Write-Host ""
Write-Host "=== Step 3. Move FAQ data (parent folder -> project) ===" -ForegroundColor Cyan

foreach ($file in @("faq_hana.jsonl", "review_flags.csv")) {
    $srcPath = Join-Path $OldDataRoot "raw\hana\faq\$file"
    $dstPath = Join-Path $ProjectRoot "data\raw\hana\faq\$file"
    if (Test-Path $srcPath) {
        Move-Item -Path $srcPath -Destination $dstPath -Force
        Write-Host "  [MOVE] $file -> data\raw\hana\faq\" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $file (not found)" -ForegroundColor DarkGray
    }
}

# ============================================================
Write-Host ""
Write-Host "=== Step 4. Copy PDFs (originals preserved in 자료실) ===" -ForegroundColor Cyan

function Copy-Pdfs($src, $dstSub, $label) {
    $dst = Join-Path $ProjectRoot $dstSub
    if (Test-Path $src) {
        $pdfs = Get-ChildItem -Path $src -Filter *.pdf -File
        foreach ($p in $pdfs) {
            Copy-Item -Path $p.FullName -Destination $dst -Force
        }
        Write-Host "  [COPY] $label $($pdfs.Count) files -> $dstSub" -ForegroundColor Green
    } else {
        Write-Host "  [NOPATH] $label source: $src" -ForegroundColor Yellow
    }
}

Copy-Pdfs $DescSrc  "data\raw\hana\desc"  "desc"
Copy-Pdfs $TermsSrc "data\raw\hana\terms" "terms"

# ============================================================
Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "File count check:" -ForegroundColor Yellow
$descN  = (Get-ChildItem "$ProjectRoot\data\raw\hana\desc"  -Filter *.pdf -File -ErrorAction SilentlyContinue).Count
$termsN = (Get-ChildItem "$ProjectRoot\data\raw\hana\terms" -Filter *.pdf -File -ErrorAction SilentlyContinue).Count
$faqOk  = Test-Path "$ProjectRoot\data\raw\hana\faq\faq_hana.jsonl"
Write-Host "  desc  PDF: $descN (expected: 80)"
Write-Host "  terms PDF: $termsN (expected: 28)"
Write-Host "  FAQ jsonl: $(if($faqOk){'OK'}else{'MISSING'})"
