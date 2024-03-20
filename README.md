# promptly

## Description
Promptly is an educational tool designed to facilitate the creation of prompts for Text-to-Image models. Currently Promplty leverages a persistent chroma vector database containing over 900k positive-negative prompt pairs. These prompt pairs have been classified by our internal prompt labeler to prevent as much toxic text as possible. The promptly system is currently 
powered by the open-source Language Model (LLM) Mistral V2 GGUF and attempts diverse prompt generation for a wide range of subjects.

The goal of this is to provide the initial framework for our community to start adding additional modalities to the website. These modalities include vector databases, prompt guided LLMs, datasets, and other systems for prompt generation.

## Features
- Access to a database of over 900k prompt pairs.
- Prompt generation based on user input context, prompt pairs from our wesite, and text2Img baseModel
- Utilizes open-source Language Model (LLM) Mistral V2 GGUF
- Diverse prompt generation meant to mimic community style 
- Tailored prompts for Text-to-Image models.

System is currently hosted on runpod but will be eventually added to the orchestrator with dedicated workers

## container
There is a container on ghcr.io with this already -> this should be attached to this repo

If you want to run the container locally, consider the following command

```
docker run -d -p 8000:8000 -e "AWS_ACCESS_KEY_ID=YOURIDKEY" -e "AWS_SECRET_ACCESS_KEY=YOURSECRETKEY3" -e "REGION_NAME=THEREGIONOFYOURS3BUCKET" -e "VECTORDBDIR=S3ADDRESSFORYOURVECTORDB" -e "DATADIR=S3ADDRESSFORADDITIONALDATA" promptly:latest  
```