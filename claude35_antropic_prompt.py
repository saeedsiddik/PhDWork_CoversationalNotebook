

import os
import pandas as pd
import json
import nbformat
from tqdm import tqdm
import time
from datetime import datetime

import anthropic


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





# Create an instance of the Anthropics API client
client = anthropic.Anthropic(api_key='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')

print ("claude 3.5 conner Anthropi")


def run_claude_model_anthropic(prompt):
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", 
                   "content": prompt}]
    )
    
    answer = response.content[0].text
    return answer


# # extar code for newly added sample
# with open("new_extra_sample_list.txt", "r") as file:
#     ipynb_files = [line.strip() for line in file]


for nb_file_name, iterate in zip(ipynb_files[2:], tqdm(range(10))):
    output_file = nb_file_name.replace('.ipynb','_claude35.csv').split('/')[-1]
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

                response = run_claude_model_anthropic(asking)

                prompt_list.append(response)
                cell_content_list.append(current_code_cell)
                cell_number_list.append(cell_number)
                # cell_type_list.append(cell_type)


        df = pd.DataFrame({'cell_number':cell_number_list, 'prompt': prompt_list, 'cell_content': cell_content_list})
        df.to_csv(output_file)
        
        with open("log_kgtorrent384_claude35.txt", "a") as file:
            file.write(formatted_time + '  done   '  + output_file + '\n')
            
        print (f"claude35 prompt done {output_file}")
        
    except Exception as e:
        
        with open("log_kgtorrent384_claude35.txt", "a") as file:
            file.write(formatted_time + '  fail   '  + output_file + str(e) + '\n')
        
        print (f"Fail claude35 prompt {output_file}")
        pass
    
    
    
print ('Execution Done')
