import torch

from lib import *
from l2_norm import L2Norm
from default_box import DefaultBox

def create_vgg():
    layers =[]
    in_channels = 3

    configs = [64, 64, 'M', 128, 128, 'M',
               256, 256, 256,'MC', 512, 512, 512,'M',
               512, 512, 512]
    for cfg in configs:
        if cfg == 'M': #floor ( làm tròn xuống)
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg =='MC': #ceiling ( làm tròn xuống)
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace= True)] #inplace= True(Không lưu) Có lưu đầu vào của hàm ReLU hay không?
            in_channels = cfg
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=3)
    layers+= [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def extras():
    layers =[]
    in_channels =1024
    config =[256, 512, 128, 256, 128, 256, 128, 256]
    layers += [nn.Conv2d(in_channels, config[0], kernel_size=1)]
    layers += [nn.Conv2d(config[0], config[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(config[1], config[2], kernel_size=1)]
    layers += [nn.Conv2d(config[2], config[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(config[3], config[4], kernel_size=1)]
    layers += [nn.Conv2d(config[4], config[5], kernel_size=3)]
    layers += [nn.Conv2d(config[5], config[6], kernel_size=1)]
    layers += [nn.Conv2d(config[6], config[7], kernel_size=3)]

    return nn.ModuleList(layers)

def create_loc_conf(num_classes=21, bbox_ratio_num = [4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers =[]
    #source1
    #loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
    #conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]

    #source2
    #loc
    loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
    #conf
    conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*num_classes, kernel_size=3, padding=1)]
    # source3
    # loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[2] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[2] * num_classes, kernel_size=3, padding=1)]
    # source4
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[3] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[3] * num_classes, kernel_size=3, padding=1)]
    # source5
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[4] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[4] * num_classes, kernel_size=3, padding=1)]
    # source6
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[5] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[5] * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)
def decode(loc, defbox_list):
    """
    :param loc: [8732, 4] (delta_x, delta_y, delta_w, delta_h)
    :param defbox: [8732, 4] (cx_d, cy_d, w_d, h_d)
    :return: boxes[xmin, ymin, xmax, ymax]
    """
    boxes = torch.cat(((defbox_list[:, :2] + 0.1*loc[:, :2])*defbox_list[:, :2]),
                      defbox_list[:, 2:] * torch.exp(loc[:, 2:]*0.2), dim=1)
    #boxes (cx, cy, w, d)
    boxes[:, :2] -= boxes[:, 2:]/2 # [cx,cy] - [w,h]/2 -> [xmin, ymin]
    boxes[:, 2:] += boxes[:, :2] # [xmin, ymin] + [w,h] -> [xmax, ymax]

    return boxes
#non-maximum supression
def nms(boxes, scores, overlap=0.45, top_k=200):
    """

    :param boxes:[num_box,4] #num_box = 8732
    :param scores: [num_box]
    :return:
    """
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    #toạ độ của các điểm x1,y1, x2, y2 trong boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2-x1, y2-y1)

    tmp_x1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(dim=0)
    idx = idx[-top_k: ]
    while idx.numel() >0:
        i = idx[-1]
        keep[count] = i
        count +=1

        if idx.size(0) ==1:
            break
        idx =idx[:-1] #idx của boxes ngoại trừ box có độ tin cậy cao nhất( idx -1)

        #infomation boxes
        torch.index_select(x1, 0, idx, out=tmp_x1)  #lấy các giá trị của x1 ở vị trí idx gán vào tmp_x1
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)


        tmp_x1 = torch.clamp(tmp_x1, x1[i]) # =x1[i] if tmp_x1 < x1[i]
        tmp_y1 = torch.clamp(tmp_y1, y1[i])
        tmp_x2 = torch.clamp(tmp_x2, x2[i])
        tmp_y2 = torch.clamp(tmp_y2, y2[i])# =y2[i] if tmp_y2 >y2[i]

        # chuyển về tensor có size tmp_w - 1(chuyển về size giống tmp_x1)
        tmp_w.resize_as_(tmp_x1)
        tmp_h.resize_as_(tmp_y1)

        tmp_w = tmp_x2 - tmp_x1 # width của phần trùng nhau
        tmp_h = tmp_y2 - tmp_y1 # height của phần trùng nhau
        tmp_w = torch.clamp(tmp_w, min=0.0) # nếu có giá trị âm thì đặt bằng 0
        tmp_h = torch.clamp(tmp_h, min=0.0)

        #area
        inter = tmp_w*tmp_h # diện tích phần trùng nhau
        rem_area = torch.index_select(area, 0, idx) # diện tích của mỗi bbox
        union = area[i] + rem_area - inter # diện tích của box keep và
        iou = inter/union

        idx = idx[iou.le(overlap)]
    return keep, count

class Detect(Function):
    def __init__(self, conf_thresh = 0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        num_batch = conf_data.size(0) # batch_size
        num_dbox = conf_data.size(1) # 8732
        num_classes = conf_data(2) # 21 classes
        conf_data = self.softmax(conf_data)
        #(num_batch, num dbox, num_classes) -> (num_classes, num_dbox, num_batch)
        conf_preds = conf_data.transpose(2, 1)

        # xử lý từng ảnh trong batch
        for  i in range(num_batch):
            # tính bounding box  từ offset information  và default box
            decode_boxes = decode(loc_data[i], dbox_list)
            # coppy confidence scores  của ảnh thứ i
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):
                c_mask = conf_preds[cl].gt(self.conf_thresh) # chỉ lấy những confidence scores > 0.01
                scores = conf_preds[cl][c_mask]

                if scores.numel() ==0:
                    continue
                l_mask = c_mask.unsquzee(1).expand(decode_boxes) # đưa chiều về giống chiều chủa decode box để tính toán



class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.num_classes = cfg["num_classes"]

        #create main module
        self.vgg = create_vgg()
        self.extras = extras()
        self.l2_norm = L2Norm()
        self.loc, self.conf = create_loc_conf(self.num_classes,cfg["bbox_aspect_num"])

        #create default box
        dbox = DefaultBox(cfg)
        self.dbox_list = dbox.create_dbox()


if __name__ == "__main__":
    cfg = {
        "num_classes": 21,  # VOC data include 20 class + 1 class backgroud
        "input_size": 300,  # SSD300
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # Trọng số cho source1 ->source6
        "feature_maps": [38, 19, 10, 5, 3, 1],
        "steps": [8, 16, 32, 64, 10, 300],  # size of default box
        "min_size": [30, 60, 111, 162, 213, 264],  # size of default box
        "max_size": [60, 111, 162, 213, 264, 315],  # size of default box
        "aspect_ratio": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # Tỉ lệ thay đổi với default box
    }

    ssd = SSD(phase="train", cfg=cfg)
    print(ssd)