import requests

def test_predict_endpoint():
    url = "http://app:5000/predict"
    payload = {
        "text": "For this purpose the Gothenburg Young Persons Empowerment Scale (GYPES) was developed."
    }
    response = requests.post(url, json=payload)
    data = response.json()
    
    assert response.status_code == 200
    assert data["success"] is True
    assert "ner_tags" in data
    assert "prediction_time" in data
    assert data["original"] == payload["text"]
    assert data["ner_tags"] == "B-O B-O B-O B-O B-LF I-LF I-LF I-LF I-LF B-O B-AC B-O B-O B-O B-O "