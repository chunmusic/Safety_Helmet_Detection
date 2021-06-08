import os
import xml.etree.ElementTree as ET
import sys

# : Batch modify the label name of the xml tag file in the VOC data set
def changelabelname(inputpath,destination):
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            file = os.path.join(inputpath,file)
            tree = ET.parse(file)
            root = tree.getroot()
            for sku in root.findall('path'):
                for file_name in root.findall('filename'):
                    sku.text = destination+str(file_name.text)
                    tree.write(file,encoding='utf-8')
        else:
            pass

if __name__ == '__main__':

    if len(sys.argv) == 3:
        inputpath = str(sys.argv[1])
        destination = str(sys.argv[2])
        changelabelname(inputpath,destination)
    
    else:
        print("Please check your syntax")
