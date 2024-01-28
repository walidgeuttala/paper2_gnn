import shutil
import os
import json
import re 
import subprocess

def find(str2, dir_path):
  # Set the directory path

  files = os.listdir(dir_path)
  file_names = [os.path.basename(file) for file in files]

  matching_names = list(filter(lambda x: re.search(str2, x, re.IGNORECASE), file_names))

  # Pass the folder name to the command line
  return matching_names[0]


number_folders = 20
data_path = 'data'
for i in range(1, number_folders+1): 
    current_folder_name = 'output{}'.format(i)
    os.rename(current_folder_name, 'output')
    files_names = os.listdir('output')
    #str1 = 'output/'+find('last', 'output/')
    str1 = 'output/'+[file for file in files_names if  "last_model_weights_trail" in file][0]
    str2 = 'output/'+[file for file in files_names if "Data_dataset_Hidden_" in file][0]
    print(str1)
    print(str2)
    #str2 = 'output/'+find('Data_', 'output/')
    #with open(str2, 'r') as f:
    #    args = json.load(f) # dodo
    #args = args['hyper-parameters']
    script = "python test_stanford_networks.py --model_weights_path {} --args_file {} --dataset_path {}".format(str1, str2, data_path)

    script = script.split()    
    subprocess.run(script)
    os.rename('output', current_folder_name)
    
