from core.detector import Detector
from core.loss import *
from utils.mask_data import MaskData
from utils.augmentations import *
from utils.metrics import AVGMetrics
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torchvision.transforms.transforms import Compose
import torch
from tqdm import tqdm
from config.mask_config import *
from config.train_config import *
from time import time
from os import path, makedirs
from datetime import datetime

class Trainer:
    def __init__(self, classes_info, model_info, size):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det = Detector(classes_info, model_info, self.device)
        # self.det.load_model("checkpoints/model_0.6161.pth")
        self.has_train_config = False
        self.size = size

    def make_train_config(self, image_path, train_path, val_path, lr, batch_size, num_workers):
        self._make_optimizer(lr)
        self._make_dataset(image_path, train_path, batch_size, val_path, num_workers)
        self._make_criterion()
        self.min_loss = float("inf")
        self.has_train_config = True

    def _make_dataset(self, image_path, train_path, batch_size, val_path=None, num_workers=6):
        train_trans = [CropResize(self.size), Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        train_trans = Compose(train_trans)
        train_data = MaskData(image_path, train_path, (self.size, self.size), train_trans)
        self.train_loader = DataLoader(train_data, batch_size, True, num_workers=num_workers, pin_memory=True)
        if val_path:
            val_trans = [CropResize(self.size), Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            val_trans = Compose(val_trans)
            val_data = MaskData(image_path, val_path, (self.size, self.size), val_trans)
            self.val_loader = DataLoader(val_data, batch_size, num_workers=num_workers, pin_memory=True)
        else:
            self.val_loader = None

    def _make_criterion(self):
        self.criterion = FocalLoss()

    def _make_optimizer(self, lr):
        self.op = Adam(self.det.get_param(), lr=lr, weight_decay=5e-4)

    def fit_one_epoch(self):
        assert self.has_train_config, "还没配置好训练参数"
        avg_loss = AVGMetrics("train_loss")
        self.det.set_status("train")
        with tqdm(self.train_loader, desc="train") as pbar:
            for data in pbar:
                self.op.zero_grad()
                data = {k: v.to(self.device) for k, v in data.items()}
                images = data["image"]
                pred = self.det.inference(images)
                loss, scalar = self.criterion(pred, data)
                avg_loss.update(loss.item(), len(images))
                loss.backward()
                self.op.step()
                pbar.set_postfix(**scalar)
        return str(avg_loss)

    def eval_one_epoch(self, save_dir, prefix=""):
        assert self.val_loader, "验证集没有构建"
        avg_loss = AVGMetrics("val_loss")
        self.det.set_status("eval")
        with torch.no_grad():
            with tqdm(self.train_loader, desc="eval") as pbar:
                for data in pbar:
                    data = {k: v.to(self.device) for k, v in data.items()}
                    images = data["image"]
                    pred = self.det.inference(images)
                    loss, scalar = self.criterion(pred, data)
                    avg_loss.update(loss.item(), len(images))
                    pbar.set_postfix(**scalar)
        if self.min_loss > avg_loss():
            self.min_loss = avg_loss()
            self.det.save_model(f"{save_dir}/{prefix}_{self.min_loss}.pth")
            print(f"min_loss update to {self.min_loss}")
        return str(avg_loss)


def run():
    today = str(datetime.fromtimestamp(time())).replace(":", ".")[:-7]
    workspace = path.join("checkpoints", today)
    if not path.exists(workspace):
        makedirs(workspace)

    trainer = Trainer(classes_info, model_info, size)
    trainer.make_train_config(images_path, train_path, val_path, lr, batch_size, num_workers)

    for epoch in range(num_epochs):
        train_log = trainer.fit_one_epoch()
        eval_log = trainer.eval_one_epoch(workspace,f"epoch={epoch}")
        print(f"epoch={epoch}")
        print(train_log)
        print(eval_log)


if __name__ == '__main__':
    run()