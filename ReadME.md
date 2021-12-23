#  SSD - Single Shot Detector 

## Data
### *VOC dataset version 2012*

## Offset
- Default box:
cx_d( center x default), cy_d, w_d( width_default), h_d(height_default)
- Transform:  delta(cx), delta(cy), delta(w), delta(h)
- Bounding box: 
+ cx= cx_d(1+ 0,1.delta(cx))
+ cy= cy_d(1+ 0,1.delta(cy))
+ w= w_d.exp(0,2.delta(w))
+ h= h_d.exp(0,2.delta(h))

## 6 Bước
* Bước 1: resize về 300x300
* Bước 2: Chuẩn bị default box(8732)- Mỗi bức ảnh được chuẩn bị 8732 default box
* Bước 3: Truyền input ảnh vào mạng SSD
Default box(8732)x (Classes:21+ 4 offset) = 218,300
* Bước 4: Lấy ra bb có confidence cao ( Lấy 200 bb cao nhất trong 8732 bd )
* Bước 5: NMS( Non-Maximum Suppression) (lấy bb có giá trị cao nhất, khử đi những thằng trùng)
* Bước 6: Chọn một threshold cho confidence
