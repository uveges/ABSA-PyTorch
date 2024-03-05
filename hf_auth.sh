#!/bin/bash

# Define your Hugging Face token
MY_HUGGINGFACE_TOKEN=""

# Install huggingface_hub
pip install huggingface_hub

# Execute the Python command
python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$MY_HUGGINGFACE_TOKEN')"

