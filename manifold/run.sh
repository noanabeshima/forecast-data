#!/bin/bash
# Simple shell script to run the Manifold data processing pipeline
# This script replicates the functionality of the Manifold scraper, filter, and analysis scripts

# Set default values
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TMP_DIR="/tmp/manifold_pipeline_${TIMESTAMP}"
FINAL_OUTPUT_DIR="."
FINAL_FILTERED_FILE="${FINAL_OUTPUT_DIR}/manifold_filtered_markets_${TIMESTAMP}.json"

RAW_FILE="${TMP_DIR}/manifold_raw_markets.json"
METRIC_FILTERED_FILE="${TMP_DIR}/manifold_metric_filtered_markets.json"
QUALITY_FILTERED_FILE="${TMP_DIR}/manifold_quality_filtered_markets.json"

# Create temporary directory
mkdir -p "$TMP_DIR"

# Function to run a command and display its output
run_command() {
    local cmd="$1"
    local description="$2"
    
    echo -e "\n================================================================================"
    echo "STEP: $description"
    echo "COMMAND: $cmd"
    echo "================================================================================"
    
    # Run the command and capture its output
    output=$(eval "$cmd" 2>&1)
    exit_code=$?
    
    # Display the output
    echo "$output"
    
    # Check if the command failed
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Command failed with return code $exit_code" >&2
        # Clean up temporary directory on error
        rm -rf "$TMP_DIR"
        exit $exit_code
    fi
}

# Step 1: Scrape data from Manifold Markets API
run_command "python scrape.py --output $RAW_FILE" "Scraping data from Manifold Markets API"

# Step 2: Apply metric-based filtering
run_command "python metric_filter.py --input $RAW_FILE --output $METRIC_FILTERED_FILE --min-volume 5481 --min-unique-bettors 20 --only-resolved --resolved-after 2023-03-01 --closed-after 2023-03-01" "Filtering markets based on metrics"

# Step 3: Apply quality-based filtering
run_command "python quality_filter.py --input $METRIC_FILTERED_FILE --output $QUALITY_FILTERED_FILE --excluded-title-patterns 'coinflip' --excluded-topics 'fun' 'manifold' 'meta-markets' 'meta-forecasting'" "Filtering markets based on quality criteria"

# Step 4: Apply LLM-based filtering
run_command "python llm_filter.py --input $QUALITY_FILTERED_FILE --output $FINAL_FILTERED_FILE" "Filtering markets using LLM"

# Print completion message
echo -e "\n================================================================================"
echo "MANIFOLD PIPELINE COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo "Filtered markets output: $FINAL_FILTERED_FILE"
echo "================================================================================" 

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TMP_DIR"
echo "Done." 
