from torch.utils.data.dataset import Dataset
from os.path import join
import cv2
import numpy as np
import math
from utils.generate_hm import boxes2hm


class MaskData(Dataset):
    def __init__(self, data_root, anno_file, input_size, transform=None):
        self.anno = self._get_anno(anno_file, data_root)
        self.transform = transform
        self.output_size = (int(input_size[0] / 4), int(input_size[1] / 4))
        self.num_classes = 2

    def _get_anno(self, anno_file, root):
        anno = []
        with open(anno_file) as f:
            for line in f.readlines():
                item = {}
                fields = line.strip().split(" ")
                name, boxes = fields[0], fields[1:]
                boxes = np.array(boxes).reshape(-1, 6)

                item["name"] = name
                item["path"] = join(root, name)
                # if(len(boxes) == 0):
                    # print(name)
                item["boxes_info"] = boxes
                anno.append(item)
        return anno

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        item = self.anno[idx]
        from PIL import Image
        try:
            img = Image.open(item["path"])
        except IOError:
            print(item["path"])
        try:
            image = np.asarray(img)[..., :3]
        except:
            print('corrupt img', item["path"])
        # image = cv2.imread(item["path"])
        bboxes = item["boxes_info"]
        data = {
            "image": image,
            "bboxes": bboxes.astype(np.float32)
        }
        if self.transform:
            data = self.transform(data)
            # self.test_show(target, result["image"], i)
        batch_hm, batch_wh, batch_reg, batch_reg_mask = boxes2hm(data["bboxes"], self.output_size, self.num_classes)
        return data["image"], batch_hm, batch_wh, batch_reg, batch_reg_mask
