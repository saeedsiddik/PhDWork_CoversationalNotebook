
import os
import pandas as pd
import json
import nbformat
from tqdm import tqdm
import time
from datetime import datetime


from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from scipy.spatial.distance import cosine
import torch

def get_filenames_with_endsWithSubString(directory_path, endsWithSubString):
    input_files = []

    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(endsWithSubString):
                # Get the full path of the file or just the file name
                input_files.append(os.path.join(root, file))

    return input_files


input_file_directory_with_prompt = '/home/jupyter-saeed3/Dataset/kgtorrent_384_prompt/'

csv_files_gpt4oMini = get_filenames_with_endsWithSubString(input_file_directory_with_prompt, 'gpt4oMini.csv')
csv_files_claude35 = get_filenames_with_endsWithSubString(input_file_directory_with_prompt, 'claude35.csv')
csv_files_codestral = get_filenames_with_endsWithSubString(input_file_directory_with_prompt, 'codestral.csv')



# Function to get embeddings from Sentence-BERT and ensure they are 1-D
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # Get mean across token embeddings and flatten to ensure 1-D vector
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()



def compare_prompt_with_sbert():
    
    # Load the tokenizer and model for Sentence-BERT
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    for claude35_file in csv_files_claude35[:5]:
        try:
            gpt4oMini_file = claude35_file.replace('_claude35.csv', '_gpt4oMini.csv')
            codestral_file = claude35_file.replace('_claude35.csv', '_codestral.csv')

            claude35_df = pd.read_csv(claude35_file)
            gpt4oMini_df = pd.read_csv(gpt4oMini_file)
            codestral_df = pd.read_csv(codestral_file)


            claude35_prompt_list = claude35_df['prompt'].tolist()
            gpt4oMini_prompt_list = gpt4oMini_df['prompt'].tolist()
            codestral_prompt_list = codestral_df['prompt'].tolist()


            cell_number_list = claude35_df['cell_number'].tolist()
            cell_content_list = claude35_df['cell_content'].tolist()


            similarity_list = []

            if (len(claude35_prompt_list) > 0 and len(gpt4oMini_prompt_list) > 0):
                if len(claude35_prompt_list) != len(gpt4oMini_prompt_list):
                    print("prompt lists are of different lengths!")

                else :
                    for i in range(len(claude35_prompt_list)):
                        prompt_from_claude35 = claude35_prompt_list[i]
                        prompt_from_gpt4oMini = gpt4oMini_prompt_list[i]

                        # Get embeddings
                        embedding_claude35 = get_embedding(prompt_from_claude35)
                        embedding_gpt4oMini = get_embedding(prompt_from_gpt4oMini)

                        # Calculate cosine similarity
                        similarity = 1 - cosine(embedding_claude35, embedding_gpt4oMini)
                        # print(f"Semantic Similarity: {similarity:.2f}")

                        similarity_list.append(similarity)

            df = pd.DataFrame({'cell_number':cell_number_list, 'similarity': similarity_list, 'claude35_prompt': claude35_prompt_list, 'gpt4oMini_prompt': gpt4oMini_prompt_list, 'cell_content': cell_content_list})
            # print (df)

            outputfile = claude35_file.replace('_claude35.csv', '_promptCompare_bert.csv')
            df.to_csv(outputfile)


        except Exception as e:
            with open("log_sample_gh_fail_compare.txt", "a") as file:
                file.write(formatted_time + '  fail   '  + claude35_file + str(e) + '\n')
            pass
        
        
        
# compare_prompt_with_sbert()
        

    
    
# Comapre prompt with llama 

# Load the model and tokenizer
model_name_llama = "meta-llama/Llama-3.2-3B"


token_hf = 'please ask me for for the token'
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama, token=token_hf)
model_llama = AutoModelForCausalLM.from_pretrained(model_name_llama, token=token_hf)

print (model_name_llama)

# Set pad_token to eos_token
if tokenizer_llama.pad_token_id is None:
    tokenizer_llama.pad_token = tokenizer_llama.eos_token
    
    
def run_llama_model_fuggingFace(prompt):
    
    # Tokenize input
    # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = tokenizer_llama(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate summary
    outputs = model_llama.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Provide attention mask
        max_new_tokens=150,  # Adjust max length of the summary
        # do_sample=False,  # Deterministic summarization
        
        do_sample=True,  # Enable sampling
        temperature=0.2,  # Adjust for randomness
        top_p=0.9         # Nucleus sampling
    )

    # Extract only the generated summary (exclude the prompt)
    generated_tokens = outputs[0]
    prompt_length = len(inputs.input_ids[0])
    answer_tokens = generated_tokens[prompt_length:]  # Exclude prompt tokens
    answer = tokenizer_llama.decode(answer_tokens, skip_special_tokens=True)
    
    return answer




def compare_prompt_with_llama():
    
    for claude35_file in csv_files_claude35[:5]:
        try:
            gpt4oMini_file = claude35_file.replace('_claude35.csv', '_gpt4oMini.csv')
            codestral_file = claude35_file.replace('_claude35.csv', '_codestral.csv')

            claude35_df = pd.read_csv(claude35_file)
            gpt4oMini_df = pd.read_csv(gpt4oMini_file)
            codestral_df = pd.read_csv(codestral_file)


            claude35_prompt_list = claude35_df['prompt'].tolist()
            gpt4oMini_prompt_list = gpt4oMini_df['prompt'].tolist()
            codestral_prompt_list = codestral_df['prompt'].tolist()


            cell_number_list = claude35_df['cell_number'].tolist()
            cell_content_list = claude35_df['cell_content'].tolist()


            similarity_list = []

            if (len(claude35_prompt_list) > 0 and len(gpt4oMini_prompt_list) > 0):
                if len(claude35_prompt_list) != len(gpt4oMini_prompt_list):
                    print("prompt lists are of different lengths!")

                else :
                    for i in range(len(claude35_prompt_list)):
                        prompt_from_claude35 = claude35_prompt_list[i]
                        prompt_from_gpt4oMini = gpt4oMini_prompt_list[i]

                        
                        asking = f"Calculate the contextual similarity between two given texts. Give me only the similarity score range from 0 to 1. Where 0 indicates no similarity, and 1 shows entirely similar meaning.\n Here is the text 1: {prompt_from_claude35} \n and here is the text 2: {prompt_from_gpt4oMini}"

                    
                        score = run_llama_model_fuggingFace(asking)

                        similarity_list.append(score)

            df = pd.DataFrame({'cell_number':cell_number_list, 'similarity_llama': similarity_list, 'claude35_prompt': claude35_prompt_list, 'gpt4oMini_prompt': gpt4oMini_prompt_list, 'cell_content': cell_content_list})
            # print (df)

            outputfile = claude35_file.replace('_claude35.csv', '_promptCompare_llama.csv')
            df.to_csv(outputfile)


        except Exception as e:
            with open("log_sample_gh_fail_compare.txt", "a") as file:
                file.write(formatted_time + '  fail   '  + claude35_file + str(e) + '\n')
            pass


    print ('Execution Done')
    
    
    
compare_prompt_with_llama()
