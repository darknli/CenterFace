from core.centernet import CenterNet_Resnet50
from torch.nn.functional import max_pool2d
from utils.common import single_class_nms
import torch
import cv2
import numpy as np

class Detector:
    def __init__(self, classes_info, model_info, device):
        self.num_classes = len(classes_info)
        self.idx2classes = classes_info
        self.model = self._model_select(model_info).to(device)
        self.device = device

    def _model_select(self, model_info):
        model_name = model_info["model_name"]
        if "resnet" in model_name.lower():
            use_pretrain = model_info["use_pretrain"] if "use_pretrain" in model_info else True
            model = CenterNet_Resnet50(self.num_classes, use_pretrain)
        else:
            raise ValueError(F"目前还没有{model_name}这个模型的实现！")
        return model

    def inference(self, images):
        images = images.to(self.device)
        pred = self.model(images)
        return pred

    def predict(self, image, size, thres=0.8, is_nms=True):
        oh, ow = image.shape[:2]
        image = image.astype(np.float32) / 255.
        image[:, :] -= (0.5, 0.5, 0.5)
        image[:, :] /= (0.5, 0.5, 0.5)
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image)
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            pred = self.inference(image)
        hm, hw, offset = pred["hm"], pred["hw"], pred["offset"]
        cv2.imshow("t", cv2.resize(hm[0, :, :, 0].cpu().numpy(), None, fx=4, fy=4))
        pool_hm = max_pool2d(hm, (15, 15), 1, 7)
        points_mask = pool_hm == hm
        maty, matx = torch.meshgrid(torch.arange(0, size//4), torch.arange(0, size//4))
        maty = maty.to(self.device)
        matx = matx.to(self.device)

        result = []
        for batch in range(len(image)):
            item = {}
            for cid in range(self.num_classes):
                obj_mask = torch.logical_and(hm[batch, :, :, cid] > thres[cid], points_mask[batch, :, :, cid])
                tmp_hw = hw[batch][obj_mask]
                tmp_offset = offset[batch][obj_mask]
                tmp_maty = maty[obj_mask]
                tmp_matx = matx[obj_mask]
                yx = tmp_offset + torch.stack([tmp_maty, tmp_matx], dim=-1)
                boxes = torch.cat([yx, tmp_hw], -1)
                boxes = boxes.cpu().numpy()[:, (1, 0, 3, 2)] * 4
                boxes = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], -1)
                boxes[:, (0, 2)] = boxes[:, (0, 2)] * ow / size
                boxes[:, (1, 3)] = boxes[:, (1, 3)] * oh /size
                boxes = np.concatenate([boxes, (hm[batch, :, :, cid][obj_mask]).cpu().numpy()[:, None]], -1)
                if is_nms:
                    boxes = single_class_nms(boxes, 0.1)
                item[cid] = boxes
            result.append(item)
        return result


    def set_status(self, status="train"):
        if status == "train":
            self.model.train()
        else:
            self.model.eval()

    def get_param(self):
        return self.model.parameters()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))