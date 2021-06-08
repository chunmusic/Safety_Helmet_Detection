#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conda install pillow
import os, glob
from PIL import Image

def main():
    filepath_list = glob.glob(input_path + '/*.png') 
    for filepath in filepath_list:
        basename  = os.path.basename(filepath)
        save_filepath = out_path + '/' + basename [:-4] + '.jpg'
        img = Image.open(filepath)
        img = img.convert('RGB')
        img.save(save_filepath, "JPEG", quality=95)
        print(filepath, '->', save_filepath)
        if flag_delete_original_files:
            os.remove(filepath)
            print('delete', filepath)

if __name__ == '__main__':
    input_path = '../dataset/images'
    out_path = input_path
    flag_delete_original_files = False
    main()