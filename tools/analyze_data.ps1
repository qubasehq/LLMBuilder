<#
.SYNOPSIS
    Analyzes and compares all data directories in the project.
.DESCRIPTION
    This script scans all data directories (raw, cleaned, processed, tokenized, etc.)
    and provides a comprehensive analysis including size comparison, file types, and more.
.NOTES
    File Name      : calculate_data_stats.ps1
    Prerequisite   : PowerShell 5.1 or later
#>

# Get the root directory of the project
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dataRoot = Join-Path $projectRoot "data"

# Define data directories to analyze
$dataDirs = @(
    @{ Name = "Raw"; Path = "raw" },
    @{ Name = "Cleaned"; Path = "cleaned" },
    @{ Name = "Processed"; Path = "processed" },
    @{ Name = "Tokenized"; Path = "tokenized" },
    @{ Name = "Vocab"; Path = "vocab" }
)

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

# Function to analyze a directory
function Get-DirectoryStats {
    param(
        [string]$dirPath,
        [string]$dirName
    )
    
    if (-not (Test-Path $dirPath)) {
        return @{
            Name = $dirName
            Exists = $false
            FileCount = 0
            TotalSize = 0
            Files = @()
        }
    }
    
    $files = Get-ChildItem -Path $dirPath -Recurse -File -Force -ErrorAction SilentlyContinue
    $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
    
    return @{
        Name = $dirName
        Exists = $true
        FileCount = $files.Count
        TotalSize = $totalSize
        Files = $files
    }
}

# Analyze all data directories
Write-Host "`n=== Analyzing Data Directories ===" -ForegroundColor Cyan
$allStats = @()

foreach ($dir in $dataDirs) {
    $dirPath = Join-Path $dataRoot $dir.Path
    Write-Host "Scanning $($dir.Name) data..." -ForegroundColor Cyan
    $stats = Get-DirectoryStats -dirPath $dirPath -dirName $dir.Name
    $allStats += $stats
}

# Display directory comparison
Write-Host "`n=== Data Directory Comparison ===" -ForegroundColor Green
$allStats | ForEach-Object {
    [PSCustomObject]@{
        'Directory' = $_.Name
        'Status' = if ($_.Exists) { 'Found' } else { 'Missing' }
        'Files' = if ($_.Exists) { $_.FileCount } else { 0 }
        'Size' = if ($_.Exists) { Format-FileSize $_.TotalSize } else { 'N/A' }
    }
} | Format-Table -AutoSize

# Show detailed analysis for each directory
foreach ($stats in $allStats | Where-Object { $_.Exists }) {
    $files = $stats.Files
    $totalSize = $stats.TotalSize
    
    Write-Host "`n=== $($stats.Name) Data Analysis ===" -ForegroundColor Green
    Write-Host ("{0,-20} {1,15:N0}" -f "Total files:", $stats.FileCount)
    Write-Host ("{0,-20} {1,15}" -f "Total size:", (Format-FileSize $stats.TotalSize))
    
    # Show breakdown by file type
    if ($files.Count -gt 0) {
        Write-Host "`n--- File Type Breakdown ---" -ForegroundColor Yellow
        $files | Group-Object Extension | 
            ForEach-Object { 
                $size = ($_.Group | Measure-Object -Property Length -Sum).Sum
                [PSCustomObject]@{
                    'Type' = if ($_.Name) { $_.Name } else { 'No Extension' }
                    'Count' = $_.Count
                    'Size' = Format-FileSize $size
                    'Percentage' = "{0:P1}" -f ($size / $totalSize)
                }
            } | Sort-Object -Property @{Expression={[long]($_.Size -replace '[^0-9.]')}} -Descending | 
            Format-Table -AutoSize
    }
    
    # Show top 5 largest files
    if ($files.Count -gt 0) {
        Write-Host "`n--- Top 5 Largest Files ---" -ForegroundColor Yellow
        $files | 
            Sort-Object -Property Length -Descending | 
            Select-Object -First 5 | 
            ForEach-Object { 
                [PSCustomObject]@{
                    'Size' = Format-FileSize $_.Length
                    'Name' = $_.Name
                    'Path' = $_.FullName.Replace($projectRoot, '...')
                }
            } | Format-Table -AutoSize
    }
}

# Calculate compression/size reduction
$rawStats = $allStats | Where-Object { $_.Name -eq 'Raw' -and $_.Exists } | Select-Object -First 1
$cleanedStats = $allStats | Where-Object { $_.Name -eq 'Cleaned' -and $_.Exists } | Select-Object -First 1

if ($rawStats -and $cleanedStats -and $rawStats.TotalSize -gt 0) {
    $reduction = 100 - (($cleanedStats.TotalSize / $rawStats.TotalSize) * 100)
    Write-Host "`n=== Data Cleaning Efficiency ===" -ForegroundColor Green
    Write-Host ("{0,-20} {1,15}" -f "Raw data size:", (Format-FileSize $rawStats.TotalSize))
    Write-Host ("{0,-20} {1,15}" -f "Cleaned data size:", (Format-FileSize $cleanedStats.TotalSize))
    Write-Host ("{0,-20} {1,15:N1}%" -f "Reduction:", $reduction)
}
