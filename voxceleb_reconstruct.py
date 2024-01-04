#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import random
import shutil


root_dir = './data/VoxCeleb/wav'


for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    

    if not os.path.isdir(subdir_path):
        continue

    count = 1

    for subsubdir in os.listdir(subdir_path):
        subsubdir_path = os.path.join(subdir_path, subsubdir)
        

        if not os.path.isdir(subdir_path):
            continue


        for file in os.listdir(subsubdir_path):
            file_path = os.path.join(subsubdir_path, file)
            

            if not file_path.endswith('.wav'):
                continue


            new_file_path = os.path.join(subdir_path, f"{subdir}_{str(count)}.WAV")

            if count > 20:
                os.remove(file_path)
            else:
                shutil.move(file_path, new_file_path)
            count = count + 1


        os.rmdir(subsubdir_path)
        
def delete_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"成功删除目录：{path}")
    else:
        print(f"目录不存在：{path}")


def delete_random_directories(directory_path, num_to_delete):
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print(f"目录不存在：{directory_path}")
        return

    directories = os.listdir(directory_path)
    random_directories = random.sample(directories, num_to_delete)

    for directory_name in random_directories:
        directory_to_delete = os.path.join(directory_path, directory_name)
        delete_directory(directory_to_delete)

# 使用示例
num_to_delete = 366 #要删除目录的个数，保证剩余500个即可
delete_random_directories(root_dir, num_to_delete)
