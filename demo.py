from core.detector import Detector
from core.loss import BaseLoss
from utils.mask_data import MaskData
from utils.augmentations import *
from utils.metrics import AVGMetrics
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torchvision.transforms.transforms import Compose
import torch
from tqdm import tqdm
from config.mask_config import *
from config.train_config import model_info

def to_image(det):
    size = 512

    val_trans = [Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    val_trans = Compose(val_trans)
    for i in range(100):

        path = f"D:/temp_data/mask/test/{i}.jpg "
        image = cv2.imread(path)
        print(image.shape)

        image = cv2.resize(image, (size, size))
        bboxes = det.predict(image.copy(), val_trans, size, 0.99)

        for cid, bbox in bboxes[0].items():
            bbox.astype(int)
            print(bbox)
            for b in bbox:
                cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("image", image)
        cv2.waitKey()

if __name__ == '__main__':
    det = Detector(classes_info, model_info, "cuda")
    det.load_model("checkpoints/model_0.89.pth")
    to_image(det)
