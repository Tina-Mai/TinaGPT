#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Set up Git configuration
git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_EMAIL"
git config --global credential.helper store # for Hugging Face CLI

# Install required Python packages
pip install -r requirements.txt

# Install and log in to Hugging Face
pip install -U "huggingface_hub[cli]"

# Auto-login to Hugging Face using the API key from .env
huggingface-cli login --token $HUGGINGFACE_API_KEY