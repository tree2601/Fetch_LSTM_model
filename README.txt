This project predicts the monthly number of receipts for 2022 using daily data from 2021.

The docker image is fetch_test:v1.2 and it contains all the files needed to run the app.

This app is packaged in a docker image, so please install Docker Desktop to proceed.

https://www.docker.com/products/docker-desktop/

Please keep the Docker Desktop open when running the app.

To get the docker image please run the following command in terminal:

docker pull luoxiaos2601/fetch_test:v1.2

Please run the following command in terminal to execute the app:

docker run -p 8501:8501 fetch_test:v1.2

Then you can see the app by entering the following url in a brower:

http://localhost:8501