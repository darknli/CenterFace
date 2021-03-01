from core.centernet import CenterNet_Resnet50
from tqdm import tqdm

class Detector:
    def __init__(self, classes_info, model_info):
        self.num_classes = len(classes_info)
        self.idx2classes = classes_info
        self.model = self._model_select(model_info)

    def _model_select(self, model_info):
        model_name = model_info["model_name"]
        if "resnet" in model_name.lower():
            use_pretrain = model_info["use_pretrain"] if "use_pretrain" in model_info else True
            model = CenterNet_Resnet50(self.num_classes, use_pretrain)
        else:
            raise ValueError(F"目前还没有{model_name}这个模型的实现！")
        return model

    def inference(self, images):
        pred = self.model(images)
        return pred

    def set_status(self, status="train"):
        if status == "train":
            self.model.train()
        else:
            self.model.eval()

    def get_param(self):
        return self.model.parameters()