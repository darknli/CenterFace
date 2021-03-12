from core.detector import Detector
from utils.augmentations import *
from torchvision.transforms.transforms import Compose
from config.mask_config import *
from config.train_config import model_info


np.random.seed(3)
colors = np.random.randint(128, 256, (100, 3))


def to_image(det):
    size = 512

    val_trans = [Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    val_trans = Compose(val_trans)
    for i in range(5, 200):

        path = f"D:/temp_data/mask/test/{i}.jpg "
        print(path)
        image = cv2.imread(path)

        image = cv2.resize(image, (size, size))
        bboxes = det.predict(image.copy(), size, (0.2, 0.2))

        for cid, bbox in bboxes[0].items():
            cls = "mask" if cid == 1 else "face"
            for b in bbox:
                prob = b[-1]
                b = b[:4].astype(int)
                cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), colors[cid].tolist(), 1, cv2.LINE_AA)
                cv2.putText(image, "{}:{}".format(cls, int(prob*100)), (b[0], b[1]), cv2.FONT_ITALIC, 1, colors[cid].tolist(), 2)
        cv2.imshow("image", image)
        cv2.waitKey()

def to_video(det):
    size = 512

    val_trans = [Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # 参数为0时调用本地摄像头；url连接调取网络摄像头；文件地址获取本地视频
    cap.set(3, 1920)  # 设置分辨率
    cap.set(4, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ret, frame = cap.read()
    while (True):
        ret, frame = cap.read()
        frame = frame[:, ::-1]
        frame = frame[:, 440: -440]
        image = cv2.resize(frame, (size, size))
        bboxes = det.predict(image.copy(), size, (0.5, 0.5))

        for cid, bbox in bboxes[0].items():
            cls = "mask" if cid == 1 else "face"
            for b in bbox:
                prob = b[-1]
                b = b[:4].astype(int)
                cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), colors[cid].tolist(), 1, cv2.LINE_AA)
                cv2.putText(image, "{}:{}".format(cls, int(prob * 100)), (b[0], b[1]), cv2.FONT_ITALIC, 1,
                            colors[cid].tolist(), 2)
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    det = Detector(classes_info, model_info, "cuda")
    det.load_model("checkpoints/2021-03-08 00.11.56/epoch=331_4.7689.pth")
    # to_image(det)
    to_video(det)
