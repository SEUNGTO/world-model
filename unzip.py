import os
import gzip
import shutil

file_list = os.listdir('data')
file_list = [os.path.join('data', f) for f in file_list]

for f in file_list : 
    
    output = f[:-3]

    with gzip.open(f, 'rb') as f_in :
        with open(output, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(f)