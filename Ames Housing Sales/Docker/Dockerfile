FROM python:3.7-slim

COPY Ames_Housing_Sales.csv ./Ames_Housing_Sales.csv

COPY train.py ./train.py
COPY inference.py ./inference.py
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt
CMD ["python", "train.py"]