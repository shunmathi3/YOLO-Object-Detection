import os
import torch
import torch.nn as nn


# compute Intersection over Union (IoU) of two bounding boxes
# the input bounding boxes are in (cx, cy, w, h) format
def compute_iou(pred, gt):
    x1p = pred[0] - pred[2] * 0.5
    x2p = pred[0] + pred[2] * 0.5
    y1p = pred[1] - pred[3] * 0.5
    y2p = pred[1] + pred[3] * 0.5
    areap = (x2p - x1p + 1) * (y2p - y1p + 1)    
   
    x1g = gt[0] - gt[2] * 0.5
    x2g = gt[0] + gt[2] * 0.5
    y1g = gt[1] - gt[3] * 0.5
    y2g = gt[1] + gt[3] * 0.5
    areag = (x2g - x1g + 1) * (y2g - y1g + 1)

    xx1 = max(x1p, x1g)
    yy1 = max(y1p, y1g)
    xx2 = min(x2p, x2g)
    yy2 = min(y2p, y2g)

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (areap + areag - inter)    
    return iou

def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    
    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids, device=output.device)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids, device=output.device)

    # Compute assignment of predicted bounding boxes for ground truth bounding boxes
    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
                if gt_mask[i, j, k] > 0:
                    # Transform ground truth box to original scale
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * grid_size
                    gt[1] = gt[1] * grid_size + j * grid_size
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size

                    select = 0
                    max_iou = -1
                    for b in range(num_boxes):
                        pred = pred_box[i, 5*b:5*b+4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou

    weight_coord = 5.0
    weight_noobj = 0.5

    loss_x = 0
    loss_y = 0
    loss_w = 0
    loss_h = 0
    loss_obj = 0
    loss_noobj = 0

    # Loop over boxes to sum losses properly
    for b in range(num_boxes):
        # For x and y, indexing into output channels for each box
        loss_x += torch.sum(box_mask[:, b] * torch.pow(gt_box[:, 0, :, :] - output[:, b*5, :, :], 2))
        loss_y += torch.sum(box_mask[:, b] * torch.pow(gt_box[:, 1, :, :] - output[:, b*5 + 1, :, :], 2))
        
        # For w and h, sqrt transform per YOLO paper
        loss_w += torch.sum(box_mask[:, b] * torch.pow(torch.sqrt(gt_box[:, 2, :, :]) - torch.sqrt(output[:, b*5 + 2, :, :]), 2))
        loss_h += torch.sum(box_mask[:, b] * torch.pow(torch.sqrt(gt_box[:, 3, :, :]) - torch.sqrt(output[:, b*5 + 3, :, :]), 2))
        
        # Objectness loss for boxes responsible for object
        loss_obj += torch.sum(box_mask[:, b] * torch.pow(box_confidence[:, b] - output[:, b*5 + 4, :, :], 2))
        
        # No object confidence loss for boxes not responsible
        loss_noobj += torch.sum((1 - box_mask[:, b]) * weight_noobj * torch.pow(output[:, b*5 + 4, :, :], 2))

    # Classification loss
    pred_cls = output[:, 5*num_boxes:, :, :]  # shape (batch_size, num_classes, 7, 7)
    gt_cls = gt_box[:, 4, :, :].long()  # assuming last channel in gt_box is class index
    
    # Convert gt_cls to one-hot vector for classification loss
    gt_cls_onehot = torch.zeros_like(pred_cls)
    for c in range(num_classes):
        gt_cls_onehot[:, c, :, :] = (gt_cls == c).float() * gt_mask

    loss_cls = torch.sum(gt_mask.unsqueeze(1) * torch.pow(pred_cls - gt_cls_onehot, 2))

    # Apply coordinate weight
    loss_x *= weight_coord
    loss_y *= weight_coord
    loss_w *= weight_coord
    loss_h *= weight_coord

    print('lx: %.4f, ly: %.4f, lw: %.4f, lh: %.4f, lobj: %.4f, lnoobj: %.4f, lcls: %.4f' %
          (loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_obj.item(), loss_noobj.item(), loss_cls.item()))

    total_loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls
    return total_loss
