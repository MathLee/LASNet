import cv2
import os
from torchvision import transforms
import numpy as np


with open(os.path.join('/home/user/EGFNet/dataset', f'all.txt'), 'r') as f:
    image_labels = f.readlines()
for i in range(len(image_labels)):
    label_path1 = image_labels[i].strip()
    imgrgb= cv2.imread('/home/user/EGFNet/dataset/seperated_images/' + label_path1 + '_rgb.png' , 0)
    imgdepth = cv2.imread('/home/user/EGFNet/dataset/seperated_images/' + label_path1 + '_th.png', 0)


    def tensor_to_PIL(tensor):
        image = tensor.squeeze(0)
        image = unloader(image)
        return image



    x1 = cv2.Sobel(imgrgb, cv2.CV_16S, 1, 0)
    y1 = cv2.Sobel(imgrgb, cv2.CV_16S, 0, 1)
    x2 = cv2.Sobel(imgdepth, cv2.CV_16S, 1, 0)
    y2 = cv2.Sobel(imgdepth, cv2.CV_16S, 0, 1)

    absX1 = cv2.convertScaleAbs(x1)
    absY1 = cv2.convertScaleAbs(y1)
    absX2 = cv2.convertScaleAbs(x2)
    absY2 = cv2.convertScaleAbs(y2)

    dst1 = cv2.addWeighted(absX1, 0.5, absY1, 0.5, 0)
    dst2 = cv2.addWeighted(absX2, 0.5, absY2, 0.5, 0)
    loader = transforms.Compose([
        transforms.ToTensor()])
    unloader = transforms.ToPILImage()



    dst1 = loader(dst1)
    dst2 = loader(dst2)
    dst = (dst1 + dst2) / 255.

    c = tensor_to_PIL(dst)
    c = np.array(c)

    cv2.imwrite('/home/user/EGFNet/dataset/edge/' + label_path1 + '.png', c)


