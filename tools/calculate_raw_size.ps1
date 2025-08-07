<#
.SYNOPSIS
    Calculates the total size of files in the data/raw directory.
.DESCRIPTION
    This script recursively scans the data/raw directory and calculates the total size
    of all files, displaying the result in a human-readable format (KB, MB, GB, or TB).
.NOTES
    File Name      : calculate_raw_size.ps1
    Prerequisite   : PowerShell 5.1 or later
#>

# Get the root directory of the project
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$rawDataPath = Join-Path $projectRoot "data\raw"

# Check if the directory exists
if (-not (Test-Path $rawDataPath)) {
    Write-Host "Raw data directory not found: $rawDataPath" -ForegroundColor Red
    exit 1
}

# Function to format file size in a human-readable format
function Format-FileSize {
    param([long]$size)
    
    $suffix = "B"
    $sizeFormatted = $size
    
    if ($size -gt 1TB) {
        $sizeFormatted = [math]::Round($size / 1TB, 2)
        $suffix = "TB"
    } elseif ($size -gt 1GB) {
        $sizeFormatted = [math]::Round($size / 1GB, 2)
        $suffix = "GB"
    } elseif ($size -gt 1MB) {
        $sizeFormatted = [math]::Round($size / 1MB, 2)
        $suffix = "MB"
    } elseif ($size -gt 1KB) {
        $sizeFormatted = [math]::Round($size / 1KB, 2)
        $suffix = "KB"
    }
    
    return "$sizeFormatted $suffix"
}

# Get all files recursively and calculate total size
Write-Host "Scanning $rawDataPath..." -ForegroundColor Cyan

$files = Get-ChildItem -Path $rawDataPath -Recurse -File -Force -ErrorAction SilentlyContinue
$totalSize = ($files | Measure-Object -Property Length -Sum).Sum
$fileCount = $files.Count

# Display results
Write-Host "`n=== Data Size Summary ===" -ForegroundColor Green
Write-Host ("{0,-20} {1,15:N0}" -f "Total files:", $fileCount)
Write-Host ("{0,-20} {1,15}" -f "Total size:", (Format-FileSize $totalSize))

# Show breakdown by file type
Write-Host "`n=== Breakdown by File Type ===" -ForegroundColor Green
$files | Group-Object Extension | 
    ForEach-Object { 
        $size = ($_.Group | Measure-Object -Property Length -Sum).Sum
        [PSCustomObject]@{
            'Extension' = if ($_.Name) { $_.Name } else { 'No Extension' }
            'Count' = $_.Count
            'Size' = Format-FileSize $size
            'Percentage' = "{0:P1}" -f ($size / $totalSize)
        }
    } | Sort-Object -Property Size -Descending | Format-Table -AutoSize

# Show largest files
Write-Host "`n=== Top 10 Largest Files ===" -ForegroundColor Green
$files | 
    Sort-Object -Property Length -Descending | 
    Select-Object -First 10 | 
    ForEach-Object { 
        [PSCustomObject]@{
            'Size' = Format-FileSize $_.Length
            'Name' = $_.Name
            'Path' = $_.FullName.Replace($projectRoot, '...')
        }
    } | Format-Table -AutoSize

# Show largest directories
Write-Host "`n=== Largest Directories ===" -ForegroundColor Green
$files | 
    Group-Object Directory | 
    ForEach-Object { 
        [PSCustomObject]@{
            'Size' = Format-FileSize (($_.Group | Measure-Object -Property Length -Sum).Sum)
            'Files' = $_.Count
            'Directory' = $_.Name.Replace($projectRoot, '...')
        }
    } | Sort-Object -Property @{Expression={[long]($_.Size -replace '[^0-9.]')}} -Descending | 
    Select-Object -First 10 | 
    Format-Table -AutoSize
