sh ./download_and_unzip_models.sh
 
 docker build -t biobert_postags . 

 docker run -v $PWD/db:/root/web_app/biobert_postags/db -p 5000:5000 biobert_postags

