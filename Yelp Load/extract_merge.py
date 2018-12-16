import json
import pandas as pd
import numpy as np
from os import system, listdir
from os.path import isfile, join


path = 'C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/JSON'

files = [f for f in listdir(path) if isfile(join(path, f))]

merged_json = []

for f in files:
    with open(f) as json_data:
      file_load = json.load(json_data)
      merged_json = merged_json + file_load
      json_data.close()

json = json.dumps(merged_json)
f = open("merged_extract.json","w")
f.write(json)
f.close()
    
      