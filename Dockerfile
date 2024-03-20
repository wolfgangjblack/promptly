FROM python:3.11

#Setup working dir
WORKDIR /app

#Copies requirements so we can install correct package
COPY ./requirements.txt ./

#copy app or main.py
COPY *.py ./

#copies utils folder with utils files
COPY utils/ ./utils/

#Adds aws root
RUN mkdir -p /root/.aws

#Set up environment variables to be used in main.py
ARG AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ARG REGION_NAME=$REGION_NAME
ARG VECTORDBDIR=$VECTORDBDIR
ARG DATADIR=$DATADIR

#install dependencies 
RUN pip install --upgrade -r /app/requirements.txt

#Expose port 8000
EXPOSE 8000

#run main.py
CMD ["python", "main.py"]