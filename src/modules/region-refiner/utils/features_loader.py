import pickle
from scipy.io import loadmat
import numpy as np
import torch

def list_features(imageset_path):
    with open(imageset_path) as f:
        ids = f.readlines()
    ids = [x.strip("\n") for x in ids]
    id_to_img_map = {k: v for k, v in enumerate(ids)}
    return id_to_img_map

def features_to_COXY(features_path, features_dictionary, min_overlap = 0.6, feat_dim = 2048):
    # features
    X = np.empty((0, feat_dim), dtype=np.float32) #np.zeros((total, feat_dim), dtype=np.float32)
    # target values
    Y = np.empty((0, 4), dtype=np.float32) #np.zeros((total, 4), dtype=np.float32)
    # overlap amounts
    O = np.empty((0), dtype=np.float32) #np.zeros((total, 1), dtype=np.float32)
    # classes
    C = np.empty((0), dtype=np.float32) #np.zeros((total, 1), dtype=np.float32)
    #cls_counts = np.zeros((num_classes, 1))
    for key in features_dictionary:
        # TODO evaluate whether to add tic_toc_print
        pth = features_path % features_dictionary[key]
        print(pth)
        if '.pkl' in pth:
            with open(pth, 'rb') as f:
                feat = pickle.load(f)
        elif '.mat' in pth:
            feat = loadmat(pth)

        gt_boxes = feat['boxes'][np.squeeze(feat['class'] > 0)]
        # add one to number of classes since python is zero-based
        gt_classes = feat['class'][np.squeeze(feat['class'] > 0)]

        max_ov = np.amax(feat['overlap'], axis=1)
        ex_boxes = feat['boxes'][max_ov > min_overlap]
        X = np.append(X, feat['feat'][max_ov > min_overlap], axis=0)
        for i in range(len(ex_boxes)):
            ex_box = ex_boxes[i]
            ov = np.empty((0), dtype=np.float32)
            for gt_box in gt_boxes:
                ov = np.append(ov, compute_overlap(gt_box, ex_box))
            max_ov = np.amax(ov)
            assignment = np.argmax(ov)
            gt_box = gt_boxes[assignment]
            # add one to number of classes since python is zero-base
            cls = gt_classes[assignment]

            src_w = ex_box[2] - ex_box[0] + np.spacing(1)
            src_h = ex_box[3] - ex_box[1] + np.spacing(1)
            src_ctr_x = ex_box[0] + 0.5 * src_w
            src_ctr_y = ex_box[1] + 0.5 * src_h

            gt_w = gt_box[2] - gt_box[0] + np.spacing(1)
            gt_h = gt_box[3] - gt_box[1] + np.spacing(1)
            gt_ctr_x = gt_box[0] + 0.5 * gt_w
            gt_ctr_y = gt_box[1] + 0.5 * gt_h

            dst_ctr_x = (gt_ctr_x - src_ctr_x) * 1 / src_w
            dst_ctr_y = (gt_ctr_y - src_ctr_y) * 1 / src_h
            dst_scl_w = np.log(gt_w / src_w)
            dst_scl_h = np.log(gt_h / src_h)

            target = np.array([[dst_ctr_x, dst_ctr_y, dst_scl_w, dst_scl_h]], dtype=np.float32)
            Y = np.append(Y, target, axis=0)
            O = np.append(O, max_ov)
            C = np.append(C, cls)

    COXY = {'C': torch.from_numpy(C).to("cuda"),
            'O': torch.from_numpy(O).to("cuda"),
            'X': torch.from_numpy(X).to("cuda"),
            'Y': torch.from_numpy(Y).to("cuda")
            }

    return COXY

def features_to_COXY_boxlist(features_path, features_dictionary, min_overlap = 0.6, feat_dim = 2048):
    # features
    X = torch.empty((0, feat_dim), dtype=torch.float32, device='cuda') #np.zeros((total, feat_dim), dtype=np.float32)
    # target values
    Y = torch.empty((0, 4), dtype=torch.float32, device='cuda') #np.zeros((total, 4), dtype=np.float32)
    # overlap amounts
    O = torch.empty((0), dtype=torch.float32, device='cuda') #np.zeros((total, 1), dtype=np.float32)
    # classes
    C = torch.empty((0), dtype=torch.float32, device='cuda') #np.zeros((total, 1), dtype=np.float32)
    #cls_counts = np.zeros((num_classes, 1))
    for key in features_dictionary:
        # TODO evaluate whether to add tic_toc_print
        pth = features_path % features_dictionary[key]
        #print(pth)
        feat = torch.load(pth)

        boxes = feat[feat.get_field('overlap') > min_overlap]
        ex_boxes = boxes.bbox
        gt_boxes = boxes.get_field('gt_bbox')
        X = torch.cat((X, boxes.get_field('features')))
        O = torch.cat((O, boxes.get_field('overlap')))
        C = torch.cat((C, boxes.get_field('classifier').type(torch.float32)))
        #print(X)

        src_w = ex_boxes[:,2] - ex_boxes[:,0] + 1
        src_h = ex_boxes[:,3] - ex_boxes[:,1] + 1
        src_ctr_x = ex_boxes[:,0] + 0.5 * src_w
        src_ctr_y = ex_boxes[:,1] + 0.5 * src_h

        gt_w = gt_boxes[:,2] - gt_boxes[:,0] + 1
        gt_h = gt_boxes[:,3] - gt_boxes[:,1] + 1
        gt_ctr_x = gt_boxes[:,0] + 0.5 * gt_w
        gt_ctr_y = gt_boxes[:,1] + 0.5 * gt_h

        dst_ctr_x = (gt_ctr_x - src_ctr_x) / src_w
        dst_ctr_y = (gt_ctr_y - src_ctr_y) / src_h
        dst_scl_w = torch.log(gt_w / src_w)
        dst_scl_h = torch.log(gt_h / src_h)

        target = torch.stack((dst_ctr_x, dst_ctr_y, dst_scl_w, dst_scl_h), dim=1)
        Y = torch.cat((Y, target), dim=0)

    COXY = {'C': C,
            'O': O,
            'X': X,
            'Y': Y
            }

    return COXY

def getIntersectionBox(box1, box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]

    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    intersectionBox = xmin, ymin, xmax, ymax
    return intersectionBox

def compute_overlap(gt, pred):
    xmin, ymin, xmax, ymax = getIntersectionBox(gt, pred)
    intersection_area = (xmax - xmin + 1) * (ymax - ymin + 1)
    if (xmax - xmin + 1) <= 0 or (ymax - ymin + 1) <= 0:
        # No overlap
        return 0
    gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    pred_area = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    overlap = intersection_area / (gt_area + pred_area - intersection_area)
    return overlap
