from utils.augmentation import Compose, ConvertFromInts,\
    ToAbsoluteCoords, PhotometricDistort,Expand, RandomSampleCrop,\
    RandomMirror, ToPercentCoords, Resize, SubtractMeans
from make_datapath import make_datapath_list
from lib import *
from extract_inform_annotation import Anno_xml
class DataTransform():
    def __init__(self, input_size, color_mean):
        self.input_size = input_size
        self.color_mean = color_mean

        self.data_transform = {
            "train": Compose([
                ConvertFromInts(), # convert image from int to float 32
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), # Change color by random
                Expand(color_mean),
                RandomSampleCrop(), # random crop image
                RandomMirror(), # xoay ngược ảnh
                ToPercentCoords(), #chuẩn hoá annotation data về [0,1]
                Resize(input_size),
                SubtractMeans(color_mean), #Subtract mean của BGR
                              ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean),
            ]),
        }
    def __call__(self, img, phase, boxes, lables):
        return self.data_transform[phase](img, boxes, lables)
if __name__ == "__main__":

    #prepare train, val
    root_path = "./data/VOCdevkit/VOC2012"
    train_image_list, train_annotation_list, val_image_list, val_annotation_list = make_datapath_list(root_path)

    #img read
    img_file_path = train_image_list[0]
    img = cv2.imread(img_file_path) # Height, width, channels
    height, width, channels = img.shape

    #annotation information
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    trans_anno = Anno_xml(classes)
    annotation_info = trans_anno(train_annotation_list[0], width, height)

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    color_mean = (104, 117, 123)
    input_size = 300

    #prepare data transform
    transform = DataTransform(input_size, color_mean)
    #transform train img
    phase = "train"
    img_tranformed, boxes, labels = transform(img,phase, annotation_info[:, :4], annotation_info[:,4])
    plt.imshow(cv2.cvtColor(img_tranformed, cv2.COLOR_BGR2RGB))
    plt.show()
    # transform train img
    phase = "val"
    img_tranformed, boxes, labels = transform(img, phase, annotation_info[:, :4], annotation_info[:, 4])
    plt.imshow(cv2.cvtColor(img_tranformed, cv2.COLOR_BGR2RGB))
    plt.show()