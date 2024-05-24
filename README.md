# NLP Group CW 2024 Surrey University

This project provides a Natural Language Processing (NLP) web application using a pre-trained BioBERT model. The application can be run in a Docker container and supports both quantized and non-quantized models.

## Installation

### Clone the repository
```bash
git clone https://github.com/MirkoSchiavone/NLP_group_cw_2024_surrey_university.git
cd NLP_group_cw_2024_surrey_university
```

### Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install the required packages
```bash
pip install -r requirements.txt
```

### Download the pre-trained model:
```bash
bash download_models.sh
```

### Create Docker Container:
```bash
docker build -t nlp-webapp .
```

### Run application in Docker:
```bash
# quantized model
docker run -e MODEL=quantized -p 5000:5000 nlp-webapp
```
or
```bash
# non-quantized model
docker run -e MODEL=biobert -p 5000:5000 nlp-webapp
```

### Deploy the application using CI/CD
```bash
bash ci_cd_pipeline.sh
```

### Test the Application
Send a POST request to the `/predict` endpoint with JSON payload:
```json
{
    "text":"For this purpose the Gothenburg Young Persons Empowerment Scale (GYPES) was developed."
}
```

Should return the following:
```json
{
    "ner_tags": "B-O B-O B-O B-O B-LF I-LF I-LF I-LF I-LF B-O B-AC B-O B-O B-O B-O ",
    "original": "For this purpose the Gothenburg Young Persons Empowerment Scale (GYPES) was developed.",
    "prediction_time": 0.717442,
    "success": true
}
