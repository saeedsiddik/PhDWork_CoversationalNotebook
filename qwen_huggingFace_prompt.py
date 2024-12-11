# Qwen Model 


from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import pandas as pd
import json
import nbformat
from tqdm import tqdm
import time
from datetime import datetime



def get_ipynb_filenames(directory_path):
    ipynb_files = []

    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.ipynb'):
                # Get the full path of the file or just the file name
                ipynb_files.append(os.path.join(root, file))

    return ipynb_files


def get_filenames_with_endsWithSubString(directory_path, endsWithSubString):
    input_files = []

    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(endsWithSubString):
                # Get the full path of the file or just the file name
                input_files.append(os.path.join(root, file))

    return input_files


def read_code_cells_with_numbers(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    code_cells = []

    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            # Combine the source lines into a single string for each code cell
            code = ''.join(cell['source'])
            # Store the cell number and code as a tuple
            code_cells.append((i+1, code))  # i+1 to make the cell number 1-based

    return code_cells


def read_all_cells(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    nb_cells = []

    for i, cell in enumerate(notebook['cells']):
            # Combine the source lines into a single string for each code cell
            code = ''.join(cell['source'])
            # Store the cell number and code as a tuple
            nb_cells.append((i+1, code, cell['cell_type']))  # i+1 to make the cell number 1-based
            
    return nb_cells


# set input data directory 
directory_path = '/home/jupyter-saeed3/Dataset/kgtorrent_sample_384/'
output_path = '/home/jupyter-saeed3/Dataset/kgtorrent_384_prompt/'

ipynb_files = get_ipynb_filenames(directory_path)

# print (len(ipynb_files))

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-Coder-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModelForCausalLM.from_pretrained(model_name, token="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")

# Set pad_token to eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    
print (model_name)

def run_collama_model_fuggingFace(prompt):
    
    # Tokenize input
    # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate summary
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Provide attention mask
        max_new_tokens=50,  # Adjust max length of the summary
        do_sample=False,  # Deterministic summarization
    )

    # Extract only the generated summary (exclude the prompt)
    generated_tokens = outputs[0]
    prompt_length = len(inputs.input_ids[0])
    answer_tokens = generated_tokens[prompt_length:]  # Exclude prompt tokens
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer

 
# # Code snippet to summarize
# previous_code_cells = """
# def fibonacci(n):
#     if n <= 0:
#         return []
#     elif n == 1:
#         return [0]
#     elif n == 2:
#         return [0, 1]
#     fib = [0, 1]
#     for i in range(2, n):
#         fib.append(fib[-1] + fib[-2])
#     return fib
# """

# current_code_cell = previous_code_cells


# # Add prompt for summarization
# prompt = "What specific prompt should I use to generate this code as the model's output? Provide only the prompt. DO NOT provide extra texts. Here is the code: \n" + current_code_cell + "\n\n\n Here is the previous part of that code snippet. Use this to understand what happerns earlier. " +  previous_code_cells


# answer = run_collama_model_fuggingFace(prompt)
# # Print the generated summary
# print("\nGenerated Summary:\n")
# print(answer)


for nb_file_name, iterate in zip(ipynb_files[:2], tqdm(range(2))):
    output_file = nb_file_name.replace('.ipynb','_qwenCoder7B.csv').split('/')[-1]
    output_file = output_path + output_file 

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time.sleep(0.05)
    
    try:
        # nb_cells = read_all_cells(nb_file_name)
        nb_cells = read_code_cells_with_numbers(nb_file_name)
        previous_code_cells = ''
        
        cell_number_list, cell_type_list, prompt_list, cell_content_list = [], [], [], []

        for cell_number, current_code_cell in nb_cells:
            if len(current_code_cell) > 0:
                
                
                asking = "What specific prompt should I use to generate this code as the model's output? Provide only the prompt. DO NOT provide extra texts. Here is the code: \n" + current_code_cell + "\n\n\n Here is the previous part of that code snippet. Use this to understand what happen earlier in the notebook. " +  previous_code_cells
                
                previous_code_cells = previous_code_cells + current_code_cell

                response = run_collama_model_fuggingFace(asking)

                prompt_list.append(response)
                cell_content_list.append(current_code_cell)
                cell_number_list.append(cell_number)
                # cell_type_list.append(cell_type)


        df = pd.DataFrame({'cell_number':cell_number_list, 'prompt': prompt_list, 'cell_content': cell_content_list})
        df.to_csv(output_file)
        
        with open("log_kgtorrent384_qwenCoder7B.txt", "a") as file:
            file.write(formatted_time + '  done   '  + output_file + '\n')
            
        print (f"qwenCoder7B prompt done {output_file}")
        
    except Exception as e:
        
        with open("log_kgtorrent384_qwenCoder7B.txt", "a") as file:
            file.write(formatted_time + '  fail   '  + output_file + str(e) + '\n')
        
        print (f"Fail qwenCoder7B prompt {output_file}")
        pass