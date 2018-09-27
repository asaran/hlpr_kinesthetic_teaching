import tarfile
from os import walk
import os
import json
import gzip
import ast 

def listdir(d):
    if not os.path.isdir(d):
        #print d
        if (d.endswith("json.gz")):
            print('extracted')
            print(d)
            with gzip.open(d, "rb") as f:
                print(f)
                data=f.readlines()
                for r in range(len(data)):
                    row = data[r]
                    data[r] = ast.literal_eval(row.strip('\n'))
                print(data)

    else:
        for item in os.listdir(d):
            listdir((d + '/' + item) if d != '/' else '/' + item)

f = []
path ="/media/akanksha/pearl_Gemini/gaze_lfd_user_study/"
for (dirpath, dirnames, filenames) in walk(path):
 print dirnames
 break


for (dirpath, dirnames, fname) in walk(path+'KT1/'):
    print fname
    break

d = path+'KT1/'
for item in os.listdir(d):
    listdir((d + '/' + item) if d != '/' else '/' + item)

if (fname.endswith("json.gz")):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()
elif (fname.endswith("tar")):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()