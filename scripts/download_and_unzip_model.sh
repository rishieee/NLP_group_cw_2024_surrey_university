#!/bin/bash

# Google Drive URL decomposition
FILE_ID="1--lamen5vIhAk6oGWiwFhzJJSdWSyLTr"
FILE_NAME="model.zip"

# Destination folder
DESTINATION_FOLDER="models"
mkdir -p ${DESTINATION_FOLDER}

# Google Drive download link
GD_URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"

# Fetching the download warning page (to get the confirm token)
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${GD_URL} -O- | sed -rn "s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p")

# Actual file download using the confirm token
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -O ${DESTINATION_FOLDER}/${FILE_NAME}

# Cleanup cookies
rm -f /tmp/cookies.txt

# Unzipping the model file
echo "Unzipping the model..."
unzip -o ${DESTINATION_FOLDER}/${FILE_NAME} -d ${DESTINATION_FOLDER}

# Remove the zip file after extraction
rm -f ${DESTINATION_FOLDER}/${FILE_NAME}

echo "Model downloaded and unzipped successfully."
