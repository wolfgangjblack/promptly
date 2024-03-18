import boto3, os, torch
import pandas as pd
from utils.utils import build_metafilter, sort_context_dict

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from cloudpathlib import CloudPath
from ctransformers import AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def get_gpu_layers() -> int:
  if torch.cuda.is_available():
    return 50

  return 0

# Retrieve environment variables
AWSKEYID = os.environ.get('AWS_ACCESS_KEY_ID')
AWSSECRET = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWSREGION = os.environ.get('REGION_NAME')
VECTORDBDIR = os.environ.get('VECTORDBDIR') 
DATADIR = os.environ.get("DATADIR")
LOCALVECTORDB = os.path.join(os.environ.get('VOLUME_PATH', '/tmp'), 'vectorDB')
LOCALDATADIR = os.path.join(os.environ.get('VOLUME_PATH', '/tmp'), 'data')


class InputConfig(BaseModel):
    basePrompt: str
    modelURI: str
    modelFile: str
    modelType: str
    temp: float
    genModelName: str
    genModelid: int
    genNSFWFilter: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

##Set up s3 connection
my_session = boto3.session.Session(
                  aws_access_key_id = AWSKEYID,
                  aws_secret_access_key = AWSSECRET,
                  region_name = AWSREGION)

s3 = my_session.resource('s3')


### Try to access VectorDB
try:
    os.mkdir(LOCALVECTORDB)
except FileExistsError:
    pass

cpVect = CloudPath(VECTORDBDIR)
cpVect.download_to(LOCALVECTORDB)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")
vectorstore = Chroma(embedding_function=embedding_function, 
                     persist_directory=LOCALVECTORDB)

### Try to access pd data
try:
    os.mkdir(LOCALDATADIR)
except FileExistsError:
    pass



cpData = CloudPath(DATADIR)
cpData.download_to(LOCALDATADIR) ##change this to something more persistant 
modeldetails = pd.read_csv(os.path.join(LOCALDATADIR,"modeldetails.csv"))[['modelName', 'baseModel', 'type', 'nsfw', 'modelId', 'strength', 'maxStrength', 'minStrength', 'trainedWords']]

sys_config = {
    'system_prompt':
    """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n""",
    }

#Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
def loadLLM(modelConfig):
    
    llm = AutoModelForCausalLM.from_pretrained(modelConfig['modelURI'],
                                           model_file=modelConfig['modelFile'], 
                                           model_type=modelConfig['modelType'],
                                           temperature = modelConfig['temp'],
                                           max_new_tokens =1500, # Max number of new tokens to generate
                                           stop = ["<|endoftext|>",
                                                   "</s>",
                                                   " ```\n    . \n ",
                                                   '\n\n\t'], # Text sequences to stop generation on
                                           context_length=4096, 
                                           gpu_layers = get_gpu_layers())
    
    return llm

@app.get("/")
async def read_root():
    return {"This works":"right now"}

def get_configs(item:InputConfig):
    basePrompt = item.model_dump()['basePrompt']
    modelURI = item.model_dump()["modelURI"]
    modelFile = item.model_dump()["modelFile"]
    modelType = item.model_dump()["modelType"]
    temp = item.model_dump()["temp"]
    genModelName = item.model_dump()["genModelName"]
    genModelid = item.model_dump()["genModelid"]
    genNSFWFilter = item.model_dump()["genNSFWFilter"]
    
    generationConfig = {'model': {'name': genModelName, 'id': genModelid},
                        'nsfw_filter': genNSFWFilter}
    
    modelConfig = {'modelURI': modelURI,
                   "modelFile": modelFile,
                   "modelType": modelType,
                   "temp": temp}

    return basePrompt, modelConfig, generationConfig

@app.post('/prompt_sim_search')
async def similarity_search_endpoint(item:InputConfig):
    
    basePrompt, modelConfig, generationConfig = get_configs(item)

    print(f"""base prompt: {basePrompt}\n modelConfig: {modelConfig}\n generationConfig: {generationConfig}\n""")
   
    llm = loadLLM(modelConfig)

    try:
        base_model = modeldetails[modeldetails['modelId'] == generationConfig['model']['id']]['baseModel'].iloc[0]
    except IndexError:
        base_model = "unk"

    generationConfig['baseModel'] = base_model

    filter_val = build_metafilter(generationConfig)

    context = vectorstore \
        .similarity_search("cleanedPrompt [TOPICKEY] "+ basePrompt,
            filter = filter_val, k = 5)
    
    sortedData = sort_context_dict(context)
    outputConfig = generationConfig
    outputConfig['basePrompt'] = basePrompt
    for i in ['pos', 'neg']:
        prompt = ""
        if i == 'pos':
            prompt_type = 'cleanedPrompt'
        else:
            prompt_type = 'negPrompt'
        prompt = f"### System:\n{sys_config['system_prompt']}\n"

        prompt += '''### Instruction:
            You are a language model tasked with expanding simple concepts to be used by advance text-to-image generation models. 
            You take in a user submitted simple prompt and are asked to add details via a series of appositive phrases. 
            Your appositive phrases should be simple ideas that expand upon the base concept by adding details to the simplified prompts.
            Only return complete sentences if the example response are also complete responses. 

            For guidance, Consider examples for similar prompts below:
                        
            ### Input: {}
            ### Response: {}
            </s>

            ### Input: {}
            ### Response: {}
            </s>

            ### Input:{}
            ### Response: {}
            </s>

            ### Input:{}
            ### Response: {}
            </s>

            ### Input:{}
            ### Response: {}
            </s>

            return a single reply similar to those above'''
        prompt += f"### Input: {basePrompt}\n### Response: "

        tmp_list = []
            
        for j in range(len(sortedData['prompt'])):
            tmp_list.append(basePrompt)
            tmp_list.append(sortedData[prompt_type][j])

        prompt = prompt.format(*tmp_list)   

        outputConfig[i+'Prompt'] = llm(prompt) # Res is a dictionaru

    return outputConfig

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host = '*', port = 8000)