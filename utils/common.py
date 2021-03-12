import numpy as np


def batch_iou(box_a, boxes):
    minx = np.maximum(box_a[0], boxes[:, 0])
    miny = np.maximum(box_a[1], boxes[:, 1])
    maxx = np.minimum(box_a[2], boxes[:, 2])
    maxy = np.minimum(box_a[3], boxes[:, 3])

    area_inter = np.maximum(maxx - minx, 0)  * np.maximum(maxy - miny, 0)
    area_a = np.maximum((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), 0)
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    iou = area_inter / (area_a + area + area_inter)
    return iou


def single_class_nms(bboxes, thres=0.7):
    result = []
    conf = bboxes[:, -1]
    rank = np.argsort(conf)[::-1]
    bboxes = bboxes[rank]
    while len(bboxes) > 0:
        result.append(bboxes[0])
        save_mask = batch_iou(bboxes[0], bboxes[1:]) < thres
        bboxes = bboxes[1:][save_mask]
    return np.array(result)
