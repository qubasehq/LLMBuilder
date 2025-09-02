"""
Example data processing tool template.
This demonstrates how to create a data processing tool.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from llmbuilder.utils.lazy_imports import pandas as pd


def process_csv(
    input_file: str,
    output_file: str,
    operations: List[str],
    columns: Optional[List[str]] = None
) -> str:
    """Process a CSV file with specified operations.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        operations: List of operations to perform (clean, dedupe, sort, filter)
        columns: Specific columns to process (optional, processes all if None)
        
    Returns:
        Summary of processing results
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read CSV file
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    original_rows = len(df)
    
    # Apply operations
    for operation in operations:
        if operation == "clean":
            df = _clean_dataframe(df, columns)
        elif operation == "dedupe":
            df = _dedupe_dataframe(df, columns)
        elif operation == "sort":
            df = _sort_dataframe(df, columns)
        elif operation == "filter":
            df = _filter_dataframe(df, columns)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    # Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    final_rows = len(df)
    
    result = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "operations": operations,
        "original_rows": original_rows,
        "final_rows": final_rows,
        "rows_removed": original_rows - final_rows,
        "columns": list(df.columns)
    }
    
    return json.dumps(result, indent=2)


def merge_csv_files(
    input_files: List[str],
    output_file: str,
    merge_type: str = "concat"
) -> str:
    """Merge multiple CSV files.
    
    Args:
        input_files: List of CSV file paths to merge
        output_file: Path to output merged CSV file
        merge_type: Type of merge (concat, inner, outer, left, right)
        
    Returns:
        Summary of merge results
    """
    dataframes = []
    
    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {file_path}: {e}")
    
    # Merge dataframes
    if merge_type == "concat":
        merged_df = pd.concat(dataframes, ignore_index=True)
    else:
        # For other merge types, merge sequentially
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, how=merge_type)
    
    # Save merged data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    result = {
        "input_files": input_files,
        "output_file": output_file,
        "merge_type": merge_type,
        "total_rows": len(merged_df),
        "total_columns": len(merged_df.columns),
        "columns": list(merged_df.columns)
    }
    
    return json.dumps(result, indent=2)


def analyze_data(
    input_file: str,
    analysis_type: str = "summary"
) -> str:
    """Analyze data in a CSV file.
    
    Args:
        input_file: Path to CSV file to analyze
        analysis_type: Type of analysis (summary, stats, missing, duplicates)
        
    Returns:
        Analysis results in JSON format
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    if analysis_type == "summary":
        result = {
            "file": str(input_path),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    elif analysis_type == "stats":
        result = {
            "file": str(input_path),
            "numeric_stats": df.describe().to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict()
        }
    elif analysis_type == "missing":
        missing_data = df.isnull().sum()
        result = {
            "file": str(input_path),
            "total_missing": missing_data.sum(),
            "missing_by_column": missing_data.to_dict(),
            "missing_percentage": (missing_data / len(df) * 100).to_dict()
        }
    elif analysis_type == "duplicates":
        duplicates = df.duplicated()
        result = {
            "file": str(input_path),
            "total_duplicates": duplicates.sum(),
            "duplicate_percentage": duplicates.sum() / len(df) * 100,
            "unique_rows": len(df) - duplicates.sum()
        }
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    return json.dumps(result, indent=2)


def convert_format(
    input_file: str,
    output_file: str,
    output_format: str
) -> str:
    """Convert data file to different format.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        output_format: Target format (json, excel, parquet, tsv)
        
    Returns:
        Conversion summary
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read input file (assume CSV for now)
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to read input file: {e}")
    
    # Convert to target format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif output_format == "excel":
        df.to_excel(output_path, index=False)
    elif output_format == "parquet":
        df.to_parquet(output_path, index=False)
    elif output_format == "tsv":
        df.to_csv(output_path, sep='\t', index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    result = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "output_format": output_format,
        "rows_converted": len(df),
        "columns_converted": len(df.columns)
    }
    
    return json.dumps(result, indent=2)


def _clean_dataframe(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Clean dataframe by removing null values and whitespace."""
    if columns:
        # Clean only specified columns
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df = df[df[col].notna()]
    else:
        # Clean all columns
        df = df.dropna()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df


def _dedupe_dataframe(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows."""
    if columns:
        return df.drop_duplicates(subset=columns)
    else:
        return df.drop_duplicates()


def _sort_dataframe(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Sort dataframe by specified columns."""
    if columns:
        # Sort by specified columns
        valid_columns = [col for col in columns if col in df.columns]
        if valid_columns:
            return df.sort_values(valid_columns)
    else:
        # Sort by first column
        if len(df.columns) > 0:
            return df.sort_values(df.columns[0])
    
    return df


def _filter_dataframe(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Filter dataframe (basic filtering - remove empty rows)."""
    if columns:
        # Filter based on specified columns not being empty
        for col in columns:
            if col in df.columns:
                df = df[df[col].notna() & (df[col] != '')]
    else:
        # Remove rows where all values are null
        df = df.dropna(how='all')
    
    return df


# Tool metadata for registration
TOOL_METADATA = {
    "name": "data_processing_tool",
    "description": "Tool for processing and analyzing CSV and other data files",
    "category": "data_processing",
    "version": "1.0.0",
    "author": "LLMBuilder Team",
    "functions": [
        "process_csv",
        "merge_csv_files",
        "analyze_data",
        "convert_format"
    ]
}