#!/bin/bash

# Set default values
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TMP_DIR="tmp_${TIMESTAMP}"
FINAL_FILE="filtered_questions.json"
RAW_FILE="${TMP_DIR}/raw_questions.json"
FILTERED_FILE="${TMP_DIR}/filtered_questions.json"

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
        # Clean up temporary directory before exiting
        rm -rf "$TMP_DIR"
        exit $exit_code
    fi
}

# Step 1: Scrape data from Metaculus API
run_command "python3 scrape.py --output $RAW_FILE --batch-size 500" "Scraping data from Metaculus API"

# Step 2: Filter data based on criteria
run_command "python3 filter.py --input $RAW_FILE --output $FILTERED_FILE --question-type binary --min-resolve-date 2023-03-01" "Filtering questions based on criteria"

# Step 3: Copy the final filtered file to the current directory
run_command "cp $FILTERED_FILE $FINAL_FILE" "Saving final filtered data to current directory"

# Clean up temporary directory
run_command "rm -rf $TMP_DIR" "Cleaning up temporary files"

# Print completion message
echo -e "\n================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo "Final output: $FINAL_FILE"
echo "================================================================================" 
