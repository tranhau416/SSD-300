from lib import *
from make_datapath import make_datapath_list


class Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes
    def __call__(self, xml_path, width, height):

        #include image annotation
        ret =[]
        # Read file xml
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            #information for bonding box
            bndbox = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            points = ['xmin', 'ymin', 'xmax', 'ymax']

            for pts in points:
                # Điểm trong dataset VOC bắt đầu từ (1,1)
                # Trong OpenCV quy ước từ (0,0)
                # Trừ đi 1 để có thể sử dụng với openCV
                pixel = int(bbox.find(pts).text) - 1
                if pts =='xmin' or pts =='xmax':
                    pixel /= width # ratio of width
                else:
                    pixel /= height
                bndbox.append(pixel)
            lable_id = self.classes.index(name)
            bndbox.append(lable_id)
            ret += [bndbox]
        return np.array(ret) #[['xmin', 'ymin', 'xmax', 'ymax','lable_id'],...['xmin', 'ymin', 'xmax', 'ymax','lable_id']]

if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    anno_xml = Anno_xml(classes)
    root_path = "./data/VOCdevkit/VOC2012"

    train_image_list, train_annotation_list, val_image_list, val_annotation_list = make_datapath_list(root_path)

    idx =1
    img_file_path = val_image_list[idx]

    print(img_file_path)
    img = cv2.imread(img_file_path)  #[height, width, channels]
    height, width, channels = img.shape #get size img
    print(f"Size image: height ={height}, width = {width}, channels = {channels}")

    #xml_path, width, height
    annotation_info = anno_xml(val_annotation_list[idx], width, height)
    print(annotation_info)