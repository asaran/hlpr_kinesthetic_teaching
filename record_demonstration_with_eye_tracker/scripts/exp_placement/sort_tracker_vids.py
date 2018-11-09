import os
import json
# from pprint import pprint
import shutil, errno

main_dir = '/home/asaran/Documents/Gaze_LfD_user_study/eye tracker SD card/projects/ded6ux7/recordings/'
save_dir = '/home/asaran/gaze_lfd_ws/src/hlpr_kinesthetic_teaching/record_demonstration_with_eye_tracker/data/eye_tracker/'

def copy_file(path):
    '''copy_file(string)

    Import the needed functions.
    Assert that the path is a file.
    Return all file data.'''
    from os.path import basename, isfile
    assert isfile(path)
    return (basename(path), file(path, 'rb', 0).read())

def paste_file(file_object, path):
    '''paste_file(tuple, string)

    Import needed functions.
    Assert that the path is a directory.
    Create all file data.'''
    from os.path import isdir, join
    assert isdir(path)
    file(join(path, file_object[0]), 'wb', 0).write(file_object[1])

def copy_dir(path):
    '''copy_dir(string)

    Import needed functions.
    Assert that path is a directory.
    Setup a storage area.
    Write all data to the storage area.
    Return the storage area.'''
    from os import listdir
    from os.path import basename, isdir, isfile, join
    assert isdir(path)
    dir = (basename(path), list())
    for name in listdir(path):
        next_path = join(path, name)
        if isdir(next_path):
            dir[1].append(copy_dir(next_path))
        elif isfile(next_path):
            dir[1].append(copy_file(next_path))
    return dir

def paste_dir(dir_object, path):
    '''paste_dir(tuple, string)

    Import needed functions.
    Assert that the path is a directory.
    Edit the path and create a directory as needed.
    Create all directories and files as needed.'''
    from os import mkdir
    from os.path import isdir, join
    assert isdir(path)
    if dir_object[0] is not '':
        path = join(path, dir_object[0])
        mkdir(path)
    for object in dir_object[1]:
        if type(object[1]) is list:
            paste_dir(object, path)
        else:
            paste_file(object, path)

folders = os.listdir(main_dir)
for f in folders:
    file = main_dir + f + '/participant.json'

    with open(file) as fi:
        data = json.load(fi)

        # read participant info
        user = data["pa_info"]["Name"]
        # print(user)

        dirName = save_dir + user
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        else:    
            print("Directory " , dirName ,  " already exists")


        # dirObj = copy_dir(main_dir+f)
        # paste_dir(dirObj, dirName)
        shutil.copytree(main_dir + f, dirName+'/'+f)
