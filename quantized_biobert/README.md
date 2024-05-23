sh ./download_and_unzip_models.sh

 docker build -t quantized_biobert . 

 docker run -v $PWD/db:/root/web_app/quantized_biobert/db -p 5000:5000 quantized_biobert

