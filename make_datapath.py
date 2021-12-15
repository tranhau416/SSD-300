from lib import *

def make_img_anno(id_names,img_path_temp, anno_path_temp):
    image_list = list()
    annotation_list = list()
    # print(id_names)
    for line in open(id_names):
        file_id = line.strip()
        img_path = (img_path_temp % file_id)
        anno_path = (anno_path_temp % file_id)
        image_list.append(img_path)
        annotation_list.append(anno_path)
    return image_list, annotation_list

def make_datapath_list( root_path):
    image_path_template = osp.join(root_path, "JPEGImages", "%s.jpg")
    annotation_path_template = osp.join(root_path, "Annotations", "%s.xml")

    train_id_names = osp.join(root_path, "ImageSets/Main/train.txt")
    val_id_names = osp.join(root_path, "ImageSets/Main/val.txt")

    train_image_list, train_annotation_list = make_img_anno(train_id_names, image_path_template, annotation_path_template)
    val_image_list, val_annotation_list = make_img_anno(val_id_names, image_path_template, annotation_path_template)
    return train_image_list, train_annotation_list, val_image_list, val_annotation_list


if __name__ == "__main__":
    root_path = "./data/VOCdevkit/VOC2012"
    train_image_list, train_annotation_list, val_image_list, val_annotation_list = make_datapath_list(root_path)
    print(len(train_image_list))
    print(train_image_list[0])