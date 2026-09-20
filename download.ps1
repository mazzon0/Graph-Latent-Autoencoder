# Windows equivalent of download.sh: downloads COCO 2017 train/val images.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$baseDir = Join-Path "data" "datasets\coco"
New-Item -ItemType Directory -Force (Join-Path $baseDir "annotations") | Out-Null
Set-Location $baseDir

foreach ($split in "train2017", "val2017") {
    if (Test-Path $split) { Write-Host "--- $split already present, skipping ---"; continue }
    Write-Host "--- Downloading $split Images ---"
    curl.exe -L -C - -o "$split.zip" "http://images.cocodataset.org/zips/$split.zip"
    if ($LASTEXITCODE -ne 0) { throw "download of $split failed" }
    Expand-Archive -Path "$split.zip" -DestinationPath . -Force
    Remove-Item "$split.zip"
}

Write-Host "--- Setup Complete ---"
Write-Host "Structure created at $(Get-Location)"
