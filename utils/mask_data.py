from torch.utils.data.dataset import Dataset
from os.path import join
import cv2
import numpy as np
import math
from utils.generate_hm import boxes2hm
import torch
from torch.nn.functional import max_pool2d


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
        batch_hm, batch_wh, batch_offset, batch_reg_mask = boxes2hm(data["bboxes"], self.output_size, self.num_classes)
        del data["bboxes"]
        data["hm"] = batch_hm
        data["hw"] = batch_wh
        data["offset"] = batch_offset
        data["mask"] = batch_reg_mask

        # hm = torch.tensor(batch_hm)
        # hw = torch.tensor(batch_wh)
        # offset = torch.tensor(batch_offset)
        # pool_hm = max_pool2d(hm, (3, 3), 1, 1)
        # points_mask = pool_hm == hm
        # maty, matx = torch.meshgrid(torch.arange(0, 512 // 4), torch.arange(0, 512 // 4))
        #
        # result = []
        # item = {}
        # for cid in range(self.num_classes):
        #     conf = hm[:, :, cid][points_mask[:, :, cid]]
        #     tmp_hw = hw[points_mask[:, :, cid]]
        #     tmp_offset = offset[points_mask[ :, :, cid]]
        #     tmp_maty = maty[points_mask[:, :, cid]]
        #     tmp_matx = matx[points_mask[:, :, cid]]
        #     obj_mask = conf > 0.99
        #     xy = tmp_offset + torch.stack([tmp_maty, tmp_matx], dim=-1)
        #     boxes = torch.cat([xy, tmp_hw], -1)
        #     boxes = boxes[obj_mask].cpu().numpy()[..., (1, 0, 3, 2)] * 4
        #     print(boxes)
        #     boxes = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], -1)
        #     boxes[:, (0, 2)] = boxes[:, (0, 2)]
        #     boxes[:, (1, 3)] = boxes[:, (1, 3)]
        #     item[cid] = boxes
        # result.append(item)
        # for cid, bbox in result[0].items():
        #     bbox = bbox.astype(int)
        #     print(bbox)
        #     for b in bbox:
        #         cv2.rectangle(data["image"], (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.imshow("image", data["image"])
        # # cv2.waitKey()
        # cv2.imshow("hm", batch_hm[..., 0])
        # cv2.waitKey()
        return data
