## script that randomly split the images into train set and test set and move them to the final output directory

import os
import random

def split_files(src_folder, dest_folder_train, dest_folder_test, split_ratio):
    files = os.listdir(src_folder)
    random.shuffle(files)
    
    split_index = int(split_ratio * len(files))
    dest1_files = files[:split_index]
    dest2_files = files[split_index:]
    
    if not os.path.exists(dest_folder_train): os.makedirs(dest_folder_train)
    if not os.path.exists(dest_folder_test): os.makedirs(dest_folder_test)
    
    for file in dest1_files:
        src_file = os.path.join(src_folder, file)
        dest_file = os.path.join(dest_folder_train, file)
        os.rename(src_file, dest_file)
        
    for file in dest2_files:
        src_file = os.path.join(src_folder, file)
        dest_file = os.path.join(dest_folder_test, file)
        os.rename(src_file, dest_file)

if __name__ == '__main__':
    for src_folder in ['artificial', 'human']:
        dest_folder_train = '../final_output_data/train/'+src_folder
        dest_folder_test = '../final_output_data/test/'+src_folder
        split_ratio = 0.7
        split_files(src_folder, dest_folder_train, dest_folder_test, split_ratio)
