#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --model <model_name> or $0 -m <model_name>"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model|-m) model_name="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Check if model_name is provided
if [ -z "$model_name" ]; then
    usage
fi

# Check if the model is available
available_models=("lera_api" "lera_baseline")

if [[ ! " ${available_models[@]} " =~ " ${model_name} " ]]; then
    echo "Model $model_name is not available. Available models: ${available_models[@]}"
    exit 1
fi


# Build docker image from given model docker file
docker_file_path="llserver/models/$model_name/dockerfile"
if [ ! -f "$docker_file_path" ]; then
    echo "Dockerfile for model $model_name not found at $docker_file_path"
    exit 1
fi

docker build -t "llmserver.$model_name" -f "$docker_file_path" .
