<<<<<<< HEAD
# BioBERT POS and NER Tagging Model

This project implements a Named Entity Recognition (NER) tagging model using BioBERT, a domain-specific variant of BERT for biomedical text.
The model is fine-tuned to perform NER tasks, integrating token embeddings with POS tag embeddings to enhance performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to provide a robust model NER tagging within the biomedical domain. The BioBERT model is used as a base, and it is extended by adding an embedding layer for POS tags. This project includes:
- A custom PyTorch model (`BioBertPosTagsClassifier`)
- A Flask-based API for serving the model
- Dockerization for easy deployment
- Basic CI/CD pipeline scripts for testing and deployment
- Comprehensive unit tests to ensure code reliability

## Model Architecture

The `BioBertPosTagsClassifier` model combines BioBERT embeddings with POS tag embeddings. The key components include:
- **BioBERT**: Pre-trained BERT model fine-tuned on biomedical text.
- **POS Embedding Layer**: Embeds POS tags which are concatenated with BioBERT embeddings.
- **Fully Connected Layer**: Processes combined embeddings to output NER tags.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or later
- PyTorch
- Transformers
- Flask
- Docker (for containerization)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/nlp-model-app.git
    cd nlp-model-app
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained model:
    ```bash
    bash download_and_unzip_models.sh
    ```

## Usage

### Running the Flask Application

1. Start the Flask server:
    ```bash
    python run.py
    ```

2. Access the API at `http://localhost:5000/predict`.

### Example API Request

Send a POST request to the `/predict` endpoint with JSON payload:
```json
{
    "text": ["This", "is", "a", "sample"],
    "pos_tags": ["DET", "VERB", "DET", "NOUN"]
}
=======
 docker build -t nlp-webapp . 

 docker run -p 5000:5000 nlp-webapp
>>>>>>> 6f067c3 (Dockerfile)
