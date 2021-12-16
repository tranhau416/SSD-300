from lib import *
from make_datapath import make_datapath_list
from transform import DataTransform
from  extract_inform_annotation import Anno_xml

class MyDataset(data.Dataset):
    def __init__(self, img_plist, anno_plist, phase, transform, anno_xml):
        self.img_plist = img_plist
        self.anno_plist = anno_plist
        self. phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_plist)

    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)

        return img, gt
    def pull_item(self, index):
        img_file_path = self.img_plist[index]
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape

        #get annotation imformation
        anno_file_path = self.anno_plist[index]
        anno_info = self.anno_xml(anno_file_path, width, height)

        # preprocessing
        img, boxes, labels = self.transform(img, self.phase, anno_info[:, :4], anno_info[:,4])

        #BGR -> RGB and (height,width,channels) -> (channels, height, width)
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        #ground truth

        gt = np.hstack((boxes, np.expand_dims(labels, axis=1))) #gt = [xmin, ymin, xmax, ymax,label]
        return img, gt, height, width
def my_collate_fn(batch):
    targets = []
    images = []

    for sample in batch:
        images.append(sample[0]) #sample[0] = img
        targets.append(torch.FloatTensor(sample[1])) #sample[1] = annotation
    #[[3, 300, 300],[3, 300, 300],...,[3, 300, 300]]
    # (batch_size ,3, 300, 300)
    images = torch.stack(images, dim=0)
    return images, targets
if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    root_path = "./data/VOCdevkit/VOC2012"
    train_image_list, train_annotation_list, val_image_list, val_annotation_list = make_datapath_list(root_path)

    color_mean = (104,117,123) #color mean voc dataset
    input_shape = 300

    train_dataset = MyDataset(train_image_list, train_annotation_list, phase="train",
                              transform = DataTransform(input_shape, color_mean), anno_xml=Anno_xml(classes))
    val_dataset = MyDataset(val_image_list, val_annotation_list, phase="val",
                              transform=DataTransform(input_shape, color_mean), anno_xml=Anno_xml(classes))
    # print(len(train_dataset))
    # print(train_dataset.__getitem__(1))

    batch_size =4
    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iter = iter(dataloader_dict["val"])
    images, targets = next(batch_iter) # get 1 sample
    print(images.size())
    print(len(targets))
    print(targets[0].size())