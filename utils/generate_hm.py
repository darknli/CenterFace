import numpy as np


def boxes2hm(boxes, size, num_classes):
    hm = np.zeros((size, size, num_classes))
    wh = np.zeros((size, size, 2))
    offset = np.zeros((size, size, 2))
    mask = np.zeros((size, size, num_classes))

    for box in boxes:
        box, cid = box[:4], box[4]
        cx, cy = (box[0] + box[2]) / 2, (box[1] +box[3]) / 2
        w, h = (box[2] - box[0]), (box[3] - box[0])
        cx_idx, cy_idx = int(cx), int(cy)
        offset[cy_idx, cx_idx] = cx - cx_idx, cy - cy_idx
        mask[cy_idx, cx_idx, cid] = 1
        get_gaussion_mat(cx_idx, cy_idx, size, size, h, w, hm[..., cid])
        wh[cy_idx, cx_idx] = 1

    return hm, wh, offset, mask


def get_gaussion_mat(cx, cy, h, w, bh, bw, hm):
    r = cal_radius(bh, bw)
    hm_x1 = max(cx - r, 0)
    hm_y1 = max(cy - r, 0)
    hm_x2 = min(cx + r + 1, w)
    hm_y2 = min(cy + r + 1, h)

    d = r * 2 + 1
    x, y = np.meshgrid(np.arange(0, d), np.arange(0, d))
    x -= r
    y -= r
    sigmma = d / 6
    gaussion_mat = np.exp(-(x**2 + y**2) /(2 * sigmma ** 2))
    gaussion_x1 = max(r - cx, 0)
    gaussion_y1 = max(r - cy, 0)
    gaussion_x2 = np.clip(d - (cx + r + 1) + w, 0, d)
    gaussion_y2 = np.clip(d - (cy + r + 1) + h, 0, d)
    hm[hm_y1: hm_y2, hm_x1: hm_y2] = gaussion_mat[gaussion_x1: gaussion_x2, gaussion_y1: gaussion_y2]


def cal_radius(h, w, overlap=0.7):
    def get_root(a, b, c):
        delta = (b**2 - 4*a*c)**0.5
        return (-b + delta) / (2 * a)

    # gt在内
    inter_r = get_root(4, -2 * (w + h), (1 - overlap) * w * h)
    # gt在外
    outer_r = get_root(4, 2 * (w + h), (1 - 1 / overlap) * w * h)
    # gt相交
    overlap_r = get_root(1, -(w + h), (1 - overlap) / (1 + overlap) * w * h)

    return max(int(min(inter_r, outer_r, overlap_r)+0.5), 1)


if __name__ == '__main__':
    hm = np.zeros((10, 10, 2))
    w, h = 18, 18
    cx, cy = 5, 5
    get_gaussion_mat(cx, cy, 10, 10, h, w, hm[..., 0])
    print(hm[..., 0])