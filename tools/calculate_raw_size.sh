#!/bin/bash

# Get the root directory of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RAW_DATA_DIR="$PROJECT_ROOT/data/raw"

# Check if the directory exists
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo -e "\033[31mRaw data directory not found: $RAW_DATA_DIR\033[0m"
    exit 1
fi

# Function to format file size in a human-readable format
format_size() {
    local size=$1
    if [ $size -ge 1099511627776 ]; then
        echo "$(echo "scale=2; $size/1099511627776" | bc) TB"
    elif [ $size -ge 1073741824 ]; then
        echo "$(echo "scale=2; $size/1073741824" | bc) GB"
    elif [ $size -ge 1048576 ]; then
        echo "$(echo "scale=2; $size/1048576" | bc) MB"
    elif [ $size -ge 1024 ]; then
        echo "$(echo "scale=2; $size/1024" | bc) KB"
    else
        echo "$size B"
    fi
}

echo -e "\033[36mScanning $RAW_DATA_DIR...\033[0m"

# Get total size and file count
echo -e "\n\033[32m=== Data Size Summary ===\033[0m"
total_size=$(find "$RAW_DATA_DIR" -type f -printf "%s\n" 2>/dev/null | awk '{total += $1} END {print total}')
total_files=$(find "$RAW_DATA_DIR" -type f 2>/dev/null | wc -l)
echo "Total files:              $total_files"
echo "Total size:               $(format_size $total_size)"

# Get breakdown by file type
echo -e "\n\033[32m=== Breakdown by File Type ===\033[0m"
find "$RAW_DATA_DIR" -type f -printf "%f\t%s\n" 2>/dev/null | awk -F. '{
    if (NF>1) { 
        ext=tolower($NF); 
        # Clean up extensions
        gsub(/[^a-z0-9]/, "", ext);
        if (length(ext) > 10) ext=substr(ext,1,10);
        if (!ext) ext="no_ext";
        ext_count[ext]++;
        ext_size[ext]+=$1;
    } else {
        ext_count["no_ext"]++;
        ext_size["no_ext"]+=$1;
    }
    total_size+=$1;
}
END {
    printf "%-15s %10s %10s %10s\n", "Extension", "Count", "Size", "% of Total"
    print "------------------------------------------------"
    for (ext in ext_count) {
        printf "%-15s %10d %10s %9.1f%%\n", 
            ext, 
            ext_count[ext], 
            sprintf("%.1f", ext_size[ext]/1024/1024) " MB",
            (ext_size[ext]/total_size)*100;
    }
}'

# Get top 10 largest files
echo -e "\n\033[32m=== Top 10 Largest Files ===\033[0m"
find "$RAW_DATA_DIR" -type f -printf "%s\t%p\n" 2>/dev/null | sort -nr | head -n 10 | awk -v root="$PROJECT_ROOT/" '{
    size=$1;
    $1="";
    path=$0;
    gsub(root, ".../", path);
    printf "%10s  %s\n", 
        sprintf("%.1f MB", size/1024/1024),
        path;
}'

# Get largest directories
echo -e "\n\033[32m=== Largest Directories ===\033[0m"
find "$RAW_DATA_DIR" -type d 2>/dev/null | while read dir; do
    size=$(find "$dir" -type f -ls 2>/dev/null | awk '{total += $7} END {print total}')
    count=$(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
    if [ "$size" != "" ] && [ $size -gt 0 ]; then
        echo "$size $count $dir"
    fi
done | sort -nr | head -n 10 | awk -v root="$PROJECT_ROOT/" '{
    size=$1;
    count=$2;
    $1=$2="";
    path=$0;
    gsub(root, ".../", path);
    printf "%10s  %6d  %s\n", 
        sprintf("%.1f MB", size/1024/1024),
        count,
        path;
}'

echo ""  # Add a newline at the end
