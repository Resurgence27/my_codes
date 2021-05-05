import torch
import torch_geometric
import torchvision
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as tfs
from torch_geometric.data import DataLoader
from pycocotools.coco import COCO


# data_set = torch_geometric.datasets.GeometricShapes("/datasets/geometric_data_script/", train=True, transform=torch_geometric.transforms.SamplePoints,pre_transform= None,pre_filter=None)
# for data in data_set:
#     print(data)
# root = "/datasets/coco"
# annFile = "/datasets/coco/annotation"
# coco_data = torchvision.datasets.CocoDetection(root, annFile, transform=torchvision.transforms.ToTensor(), target_transform=None, transforms=None)


# cv implementt
# former_pic = cv2.imread("F:/MyCode/PyCharm/pic/a.jpg")
# former_pic=cv2.cvtColor(former_pic,cv2.COLOR_BGR2GRAY)
# # cv2.imshow("former_pic",former_pic)
# theta = 15
# M_rotate = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta), np.cos(theta), 0]
# ], dtype=np.float32)
img_rotated = cv2.warpAffine(former_pic, M_rotate, (300, 200))
# cv2.imshow("latter_pic",img_rotated)
# cv2.getAffineTransform(former_pic,img_rotated)

former_pic = Image.open("F:/MyCode/PyCharm/pic/a.jpg")
Image._show(former_pic)

corp_size = 200
degree = (0,90)
translation = (0,0.2)
scale = (0.8, 1)

trans_com = tfs.Compose([
 tfs.CenterCrop(corp_size),
 tfs.RandomAffine(degrees=degree, translate=translation, scale=scale),  # include translation, scale, rotation
 tfs.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0)
])
ttransform_pic = trans_com(former_pic)

d,t,s,shear = tfs.RandomAffine.get_params(degree,translation,scale,(0,0),(180,200))
Image._show(ttransform_pic)