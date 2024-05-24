#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the Docker image name
IMAGE_NAME="nlp-webapp"

# Step 1: Build the Docker image
echo "======================="
echo " BUILDING DOCKER IMAGE "
echo "======================="
docker build -t $IMAGE_NAME .

# Step 2: Run unit tests
echo "===================="
echo " RUNNING UNIT TESTS "
echo "===================="
docker run --rm $IMAGE_NAME pytest test_unittest.py -vv

# Step 3: Run integration tests
echo "============================================="
echo " RUNNING INTEGRATION TESTS (QUANTIZED MODEL) "
echo "============================================="
MODEL=quantized docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

echo "==========================================="
echo " RUNNING INTEGRATION TESTS (BIOBERT MODEL) "
echo "==========================================="
MODEL=biobert docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

# Step 4: Deploy the Application
echo "======================="
echo " DEPLOYING APPLICATION "
echo "======================="
docker run -e MODEL=quantized -p 5000:5000 nlp-webapp