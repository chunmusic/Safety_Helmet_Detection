import os
import xml.etree.ElementTree as ET
import sys

# : Batch modify the label name of the xml tag file in the VOC data set
def changelabelname(inputpath):
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            file = os.path.join(inputpath,file)
            tree = ET.parse(file)
            root = tree.getroot()
            for sku in root.findall('filename'):

                name, surname = sku.text.split('.')

                new_surname = ".jpg"

                sku.text = name+new_surname
                tree.write(file,encoding='utf-8')

        else:
            pass

if __name__ == '__main__':

    if len(sys.argv) == 2:
        inputpath = str(sys.argv[1])
        changelabelname(inputpath)
    
    else:
        print("Please check your syntax")
