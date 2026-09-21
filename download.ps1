# Windows equivalent of download.sh: downloads COCO 2017 and/or CLEVR v1.0.
# Usage: powershell -ExecutionPolicy Bypass -File download.ps1 [coco|clevr|all]   (default: coco)
param([ValidateSet("coco", "clevr", "all")][string]$Dataset = "coco")

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root = (Get-Location).Path

function Download-File($url, $out) {
    # curl.exe supports resuming with -C -; it resumes by itself if the connection drops
    curl.exe -L -C - --retry 50 --retry-delay 5 --retry-all-errors -o $out $url
    if ($LASTEXITCODE -ne 0) { throw "download of $url failed" }
}

# Extracts only the entries of a zip whose path starts with one of the given prefixes
function Expand-ZipPrefixes($zipPath, $destDir, $prefixes) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName.Replace("\", "/")
            if ($entry.Name -eq "" -or -not ($prefixes | Where-Object { $name.StartsWith($_) })) { continue }
            $target = Join-Path $destDir $name
            New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
        }
    } finally { $zip.Dispose() }
}

function Download-Coco {
    $baseDir = Join-Path $root "data\datasets\coco"
    New-Item -ItemType Directory -Force (Join-Path $baseDir "annotations") | Out-Null
    Set-Location $baseDir
    foreach ($split in "train2017", "val2017") {
        if (Test-Path $split) { Write-Host "--- $split already present, skipping ---"; continue }
        Write-Host "--- Downloading $split Images ---"
        Download-File "http://images.cocodataset.org/zips/$split.zip" "$split.zip"
        Expand-Archive -Path "$split.zip" -DestinationPath . -Force
        Remove-Item "$split.zip"
    }
    Set-Location $root
    Write-Host "--- COCO Setup Complete: $baseDir ---"
}

function Download-Clevr {
    $baseDir = Join-Path $root "data\datasets\clevr"
    New-Item -ItemType Directory -Force $baseDir | Out-Null
    Set-Location $baseDir
    if (Test-Path "CLEVR_v1.0\images\val") { Write-Host "--- CLEVR already present, skipping ---"; Set-Location $root; return }

    Write-Host "--- Downloading CLEVR v1.0 (about 19 GB) ---"
    Download-File "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip" "CLEVR_v1.0.zip"
    Write-Host "--- Extracting ---"
    Expand-ZipPrefixes (Join-Path $baseDir "CLEVR_v1.0.zip") $baseDir @("CLEVR_v1.0/images/train/", "CLEVR_v1.0/images/val/", "CLEVR_v1.0/scenes/")
    Remove-Item "CLEVR_v1.0.zip"
    Set-Location $root
    Write-Host "--- CLEVR Setup Complete: $baseDir\CLEVR_v1.0 ---"
}

if ($Dataset -in "coco", "all")  { Download-Coco }
if ($Dataset -in "clevr", "all") { Download-Clevr }
