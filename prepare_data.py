import os
import  urllib.request
import  zipfile
import tarfile

data_dir  = "./data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
target_part= os.path.join(data_dir, "VOCtrainval_11-May-2012")

if not os.path.exists(target_part):
    urllib.request.urlretrieve(url, target_part)

    tar = tarfile.TarFile(target_part)
    tar.extractall(data_dir)
    tar.close()
