from core.centernet import CenterNet_Resnet50
from torch.nn.functional import max_pool2d
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

    def predict(self, image, trans, size, thres=0.8):
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
        cv2.imshow("t", hm[0, :, :, 0].cpu().numpy())
        cv2.waitKey()
        pool_hm = max_pool2d(hm, (3, 3), 1, 1)
        points_mask = pool_hm == hm
        maty, matx = torch.meshgrid(torch.arange(0, size//4), torch.arange(0, size//4))
        maty = maty.to(self.device)
        matx = matx.to(self.device)

        result = []
        for batch in range(len(image)):
            item = {}
            for cid in range(self.num_classes):
                conf = hm[batch, :, :, cid][points_mask[batch, :, :, cid]]
                tmp_hw = hw[batch][points_mask[batch, :, :, cid]]
                tmp_offset = offset[batch][points_mask[batch, :, :, cid]]
                tmp_maty = maty[points_mask[batch, :, :, cid]]
                tmp_matx = matx[points_mask[batch, :, :, cid]]
                obj_mask = conf > thres
                xy = tmp_offset[:, (1, 0)] + torch.stack([tmp_matx, tmp_maty], dim=-1)
                boxes = torch.cat([xy, tmp_hw], -1)
                boxes = boxes[obj_mask].cpu().numpy()[:, (0, 1, 3, 2)] * 4
                boxes = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], -1)
                # boxes = boxes / 512 * size
                boxes[:, (0, 2)] = boxes[:, (0, 2)] * ow / size
                boxes[:, (1, 3)] = boxes[:, (1, 3)] * oh /size
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