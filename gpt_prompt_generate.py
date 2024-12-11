
import os
import pandas as pd
from openai import OpenAI
import json
import openai
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

print (len(ipynb_files))



# local_drive_path = "/Volumes/LaCie/JupyterLabBackup/ChatGPT_in_notebook/"
# openai.api_key = open(local_drive_path+"Chatgpt_OpenAI_Key.txt").readline().rstrip()

jupyter_lab_path = "/home/jupyter-saeed3/ChatGPT_in_notebook/"
openai.api_key = open(jupyter_lab_path+"Chatgpt_OpenAI_Key.txt").readline().rstrip()


client = OpenAI(
    api_key=openai.api_key,
)
print (client)





def chat_with_gpt(prompt, role="user", temperature=0.2):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": role,
                "content": prompt,
            }
        ],
        # model="gpt-3.5-turbo",
        # model="gpt-4o",
        # model="gpt-4",
        model="gpt-4o-mini",
        temperature = temperature,
    )
    
    return chat_completion.choices[0].message.content


def process_gpt_query(messages):
    print (messages[0])
    

# extar code for newly added sample
with open("new_extra_sample_list.txt", "r") as file:
    ipynb_files = [line.strip() for line in file]



for nb_file_name, iterate in zip(ipynb_files[1:], tqdm(range(11))):
    output_file = nb_file_name.replace('.ipynb','_gpt4oMini.csv').split('/')[-1]
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
                
                # use this in the previous version
                # previous_code_cells = previous_code_cells + current_code_cell
                # messages = "What specific prompt should I use to generate this code as the model's output?  Here is the code: \n\n " + current_code_cell
                
                # asking = "Your job is to provide the prompt that should be used to generate the given code snippet. Answer me only the prompt for this code. \n" + current_code_cell + "\n\n\n Here is the previous part of that code snippet. Use this to understand what happerns earlier. " +  previous_code_cells
                
                asking = "What specific prompt should I use to generate this code as the model's output? Provide only the prompt. DO NOT provide extra texts. Here is the code: \n" + current_code_cell + "\n\n\n Here is the previous part of that code snippet. Use this to understand what happerns earlier. " +  previous_code_cells
                
                previous_code_cells = previous_code_cells + current_code_cell

                response = chat_with_gpt(asking)

                prompt_list.append(response)
                cell_content_list.append(current_code_cell)
                cell_number_list.append(cell_number)
                # cell_type_list.append(cell_type)


        df = pd.DataFrame({'cell_number':cell_number_list, 'prompt': prompt_list, 'cell_content': cell_content_list})
        df.to_csv(output_file)
        
        with open("log_sample_gh_gpt40mini.txt", "a") as file:
            file.write(formatted_time + '  done   '  + output_file + '\n')
            
        print (f"Gpt prompt done {output_file}")
        
    except Exception as e:
        
        with open("log_sample_gh_gpt_40mini.txt", "a") as file:
            file.write(formatted_time + '  fail   '  + output_file + str(e) + '\n')
        
        print (f"Fail Gpt prompt done {output_file}")
        pass