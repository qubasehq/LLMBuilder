#!/bin/bash

# Get the root directory of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$PROJECT_ROOT/data"

# Define data directories to analyze
declare -A DATA_DIRS=(
    ["Raw"]="raw"
    ["Cleaned"]="cleaned"
    ["Processed"]="processed"
    ["Tokenized"]="tokenized"
    ["Vocab"]="vocab"
)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to format file size in a human-readable format
format_size() {
    local size=$1
    if [ $size -ge 1099511627776 ]; then
        printf "%.2f TB" $(echo "scale=2; $size/1099511627776" | bc)
    elif [ $size -ge 1073741824 ]; then
        printf "%.2f GB" $(echo "scale=2; $size/1073741824" | bc)
    elif [ $size -ge 1048576 ]; then
        printf "%.2f MB" $(echo "scale=2; $size/1048576" | bc)
    elif [ $size -ge 1024 ]; then
        printf "%.2f KB" $(echo "scale=2; $size/1024" | bc)
    else
        echo "$size B"
    fi
}

# Function to analyze a directory
get_dir_stats() {
    local dir_path="$1"
    local dir_name="$2"
    
    if [ ! -d "$dir_path" ]; then
        echo "$dir_name 0 0"
        return
    fi
    
    local file_count=$(find "$dir_path" -type f 2>/dev/null | wc -l)
    local total_size=0
    
    if [ $file_count -gt 0 ]; then
        total_size=$(find "$dir_path" -type f -printf "%s\n" 2>/dev/null | awk '{total += $1} END {print total}')
        total_size=${total_size:-0}
    fi
    
    echo "$dir_name $file_count $total_size"
}

# Function to get file type breakdown
get_file_types() {
    local dir_path="$1"
    
    find "$dir_path" -type f -printf "%f\t%s\n" 2>/dev/null | awk -F. '{
        if (NF>1) { 
            ext=tolower($NF); 
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
        if (total_size > 0) {
            printf "%-15s %10s %15s %10s\n", "Type", "Count", "Size", "% of Total"
            print "------------------------------------------------"
            for (ext in ext_count) {
                printf "%-15s %10d %15s %9.1f%%\n", 
                    ext, 
                    ext_count[ext], 
                    sprintf("%.2f MB", ext_size[ext]/1048576),
                    (ext_size[ext]/total_size)*100;
            }
        }
    }'
}

# Function to get largest files
get_largest_files() {
    local dir_path="$1"
    local limit=${2:-5}
    
    find "$dir_path" -type f -printf "%s\t%p\n" 2>/dev/null | 
    sort -nr | 
    head -n $limit | 
    awk -v root="$PROJECT_ROOT/" '{
        size=$1;
        $1="";
        path=$0;
        gsub(root, ".../", path);
        printf "%12s  %s\n", 
            sprintf("%.2f MB", size/1048576),
            path;
    }'
}

# Main execution
echo -e "${CYAN}=== Analyzing Data Directories ===${NC}"

# Get stats for all directories
declare -A dir_stats
for dir_name in "${!DATA_DIRS[@]}"; do
    dir_path="$DATA_ROOT/${DATA_DIRS[$dir_name]}"
    echo -e "${CYAN}Scanning $dir_name data...${NC}"
    read name count size < <(get_dir_stats "$dir_path" "$dir_name")
    dir_stats["$name"]="$count $size"
done

# Display directory comparison
echo -e "\n${GREEN}=== Data Directory Comparison ===${NC}"
printf "%-15s %10s %15s %10s\n" "Directory" "Status" "Files" "Size"
echo "------------------------------------------------"
for dir_name in "${!DATA_DIRS[@]}"; do
    read count size <<< "${dir_stats[$dir_name]}"
    if [ -d "$DATA_ROOT/${DATA_DIRS[$dir_name]}" ]; then
        printf "%-15s %10s %15d %15s\n" \
            "$dir_name" \
            "Found" \
            "$count" \
            "$(format_size $size)"
    else
        printf "%-15s %10s %15s %15s\n" \
            "$dir_name" \
            "Missing" \
            "0" \
            "N/A"
    fi
done | sort

# Show detailed analysis for each directory
for dir_name in "${!DATA_DIRS[@]}"; do
    dir_path="$DATA_ROOT/${DATA_DIRS[$dir_name]}"
    read count size <<< "${dir_stats[$dir_name]}"
    
    if [ ! -d "$dir_path" ] || [ "$count" -eq 0 ]; then
        continue
    fi
    
    echo -e "\n${GREEN}=== $dir_name Data Analysis ===${NC}"
    printf "%-20s %15d\n" "Total files:" "$count"
    printf "%-20s %15s\n" "Total size:" "$(format_size $size)"
    
    # Show breakdown by file type
    echo -e "\n${YELLOW}--- File Type Breakdown ---${NC}"
    get_file_types "$dir_path"
    
    # Show top 5 largest files
    echo -e "\n${YELLOW}--- Top 5 Largest Files ---${NC}"
    get_largest_files "$dir_path" 5
    
done

# Calculate and show data cleaning efficiency
if [ -d "$DATA_ROOT/raw" ] && [ -d "$DATA_ROOT/cleaned" ]; then
    raw_size=$(find "$DATA_ROOT/raw" -type f -printf "%s\n" 2>/dev/null | awk '{total += $1} END {print total}')
    cleaned_size=$(find "$DATA_ROOT/cleaned" -type f -printf "%s\n" 2>/dev/null | awk '{total += $1} END {print total}')
    
    if [ "$raw_size" -gt 0 ] && [ "$cleaned_size" -gt 0 ]; then
        reduction=$(echo "scale=2; (1 - $cleaned_size / $raw_size) * 100" | bc)
        echo -e "\n${GREEN}=== Data Cleaning Efficiency ===${NC}"
        printf "%-20s %15s\n" "Raw data size:" "$(format_size $raw_size)"
        printf "%-20s %15s\n" "Cleaned data size:" "$(format_size $cleaned_size)"
        printf "%-20s %15s%%\n" "Reduction:" "$reduction"
    fi
fi

echo ""  # Add a newline at the end
