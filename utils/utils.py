def build_metafilter(input_config: dict) -> dict:
    """
    This function uses an input config to design a filter recognized by
    the chroma vector database. The filter is a json type meant to have
    multiple dictionaries bounded by the logic type.
    
    Note: All metadata filters MUST have present fields in the vectorDB

    Input config has:
        baseModel: Something like SDXL or SD1.5 
            new models may show up as 'unk'
        nsfw_filter: this is the users nsfw filter and is either 'safe' or ''
        tag: CURRENTLY UNSUPPORTED 
        
    """
    meta_filter = {}

    # Define model condition
    if input_config['baseModel'] != 'unk':
        model_cond = {"baseModel": input_config["baseModel"]}
    

    # Check if nsfw_filter is safe
    if input_config.get('nsfw_filter') == 'safe':
        meta_filter.setdefault("$and", []).append({"nsfw": "safe"})

    # Check if desired_tag is specified
    desired_tag = input_config.get('desired_tag')
    if desired_tag:
        tag_cond = {'tag': desired_tag}
        meta_filter.setdefault("$and", []).append(tag_cond)

    # Append model condition to meta_filter
    if meta_filter != {}:
        meta_filter.setdefault("$and", []).append(model_cond)
    else: return model_cond
    
    return meta_filter

def sort_context_dict(context):
    """
    Helper function to take context from vectorDB and generate a sorted dictionary of prompts, cleaned prompts, and negative prompts
    This is a little fragile and must be tailored to the specific vectorDB
    """
    result_dict = {}

    for doc in context:
        page_con_list = doc.page_content.split('[USEROWSEP]')
        for i in page_con_list:
            k,v = i.split('[TOPICKEY]')
            k = k[:-1]
            v = v[1:] #remove white spaces
            if 'prompt' not in k:
                v = v.replace('\n','')
            try:
                v = float(v)  # Attempt to convert v to float
            except ValueError:
                pass  # If conversion fails, v remains a string
            if k in result_dict.keys():
                result_dict[k].append(str(v).replace('\n',''))
            else:
                result_dict[k] = [v]
    # Zip the lists together
    zipped_lists = zip(result_dict['prompt'], 
                        result_dict['cleanedPrompt'],
                        result_dict['cleanedNegPrompt']
                        )

    sorted_lists = sorted(zipped_lists, key=lambda x: x[0])

    sortedPrompt, sortedCleanedPrompt, sortedCleanedNegPrompt = zip(*sorted_lists)

    # Create a new sorted dictionary
    sorted_data = {'prompt': list(sortedPrompt),
                'cleanedPrompt': list(sortedCleanedPrompt),
                "negPrompt": list(sortedCleanedNegPrompt)
                }
    
    return sorted_data

def generate_prompt(system_prompt: list,
                    context: list,
                    input_prompt: list) -> str:
    """
    An unused function broken out in main.py for readability and debugging
    This is the prompt template used in EDA and testing to generate prompts
    with Mistral
    """
    prompt = f"### System:\n{system_prompt}\n"

    prompt += """### Instruction:
    You are a language model tasked with expanding simple concepts to be used by advance text-to-image generation models. 
    You take in a user submitted simple prompt and are asked to add details via a series of appositive phrases. 
    Your appositive phrases should be simple ideas that expand upon the base concept by adding details to the simplified prompts.
    Only return complete sentences if the example response are also complete responses. 
    For guidance, Consider examples for similar prompts below:
    
    ```
    ### Input: {}
    ### Response: {}

    ### Input: {}
    ### Response: {}

    ### Input:{}
    ### Response: {}
    ```

    Use the above to expand the simplified prompt below:
    """

    sorted_data = sort_context_dict(context)
    tmp_list = []
    
    for i in range(len(sorted_data['prompt'])):
        tmp_list.append(sorted_data['cleanedPrompt '][i])
        tmp_list.append(sorted_data['prompt'][i])
    
    prompt = prompt.format(*tmp_list)   
    prompt += f"### Input: {input_prompt}\n### Response: "

    return prompt
