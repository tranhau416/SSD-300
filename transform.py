from utils.augmentation import Compose, ConvertFromInts,\
    ToAbsoluteCoords, PhotometricDistort,Expand, RandomSampleCrop,\
    RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform():
    def __init__(self, input_size, color_mean):
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
            "val": Compose,
        }
        self.input_size = input_size
        self.color_mean = color_mean
