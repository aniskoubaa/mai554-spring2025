#!/bin/bash

# Set the default output directory
OUTPUT_DIR="./output"

# Function to display usage information
function show_usage {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL         Translation model to use (default: Helsinki-NLP/opus-mt-en-fr)"
    echo "  -s, --source LANG         Source language code (default: en)"
    echo "  -t, --target LANG         Target language code (default: fr)"
    echo "  -n, --num NUM             Number of examples to translate (default: 10)"
    echo "  -i, --input TEXT          Single text to translate (enclose in quotes)"
    echo "  -o, --output DIR          Output directory (default: ./output)"
    echo "  -v, --verbose             Enable verbose output"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --input \"Hello world\" --source en --target fr"
    echo "  $0 --model Helsinki-NLP/opus-mt-en-de --source en --target de --num 5"
    echo ""
}

# Parse command line arguments
POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -m|--model)
            MODEL="$2"
            shift
            shift
            ;;
        -s|--source)
            SOURCE="$2"
            shift
            shift
            ;;
        -t|--target)
            TARGET="$2"
            shift
            shift
            ;;
        -n|--num)
            NUM_EXAMPLES="$2"
            shift
            shift
            ;;
        -i|--input)
            INPUT_TEXT="$2"
            shift
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

# Set default values if not provided
MODEL=${MODEL:-"Helsinki-NLP/opus-mt-en-fr"}
SOURCE=${SOURCE:-"en"}
TARGET=${TARGET:-"fr"}
NUM_EXAMPLES=${NUM_EXAMPLES:-"10"}
VERBOSE=${VERBOSE:-""}

# Construct the Python command
PYTHON_CMD="python3 machine_translation_demo.py --model $MODEL --source $SOURCE --target $TARGET --num_examples $NUM_EXAMPLES --output_dir $OUTPUT_DIR $VERBOSE"

# Add input text if provided
if [ ! -z "$INPUT_TEXT" ]; then
    PYTHON_CMD="$PYTHON_CMD --input_text \"$INPUT_TEXT\""
fi

# Echo the command before running
echo "Running: $PYTHON_CMD"

# Execute the command
eval $PYTHON_CMD

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Translation failed."
    exit 1
fi

echo "Translation completed successfully!" 