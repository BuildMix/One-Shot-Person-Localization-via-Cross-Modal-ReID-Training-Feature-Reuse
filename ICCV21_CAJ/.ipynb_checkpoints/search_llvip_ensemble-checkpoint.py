import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from model import embed_net

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

def load_network(checkpoint_path):
    model = embed_net(class_num=395, no_local='on', gm_pool='on', arch='resnet50')
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model

def get_feature_l2_l3(model, img_tensor, mode='visible'):
    """分别提取 Layer 2 和 Layer 3 的特征"""
    with torch.no_grad():
        x = model.visible_module(img_tensor) if mode == 'visible' else model.thermal_module(img_tensor)
        x = model.base_resnet.base.layer1(x)
        feat_l2 = model.base_resnet.base.layer2(x) # Stride 8
        feat_l3 = model.base_resnet.base.layer3(feat_l2) # Stride 16
    return feat_l2, feat_l3

def main():
    MODEL_PATH = 'save_model/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.01_seed_0_best.t' 
    JSON_PATH = 'llvip_test_pairs.json'
    RESULT_DIR = 'search_results_ensemble'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_network(MODEL_PATH)
    with open(JSON_PATH, 'r') as f:
        data_pairs = json.load(f)

    all_ious = []
    print(f"开始结果融合实验 (Layer 2 * Layer 3)...")
    
    for i, item in enumerate(tqdm(data_pairs)):
        try:
            vis_img = Image.open(item['vis_path']).convert('RGB')
            ir_img = Image.open(item['ir_path']).convert('RGB')
            tx, ty, tw, th = item['bbox']
            
            # 1. 准备数据
            template = vis_img.crop((tx, ty, tx + tw, ty + th))
            temp_t = common_transform(template).unsqueeze(0).cuda()
            ir_t = common_transform(ir_img).unsqueeze(0).cuda()
            
            # 2. 分别提取特征
            # Layer 2 用于精确定位，Layer 3 用于语义确认
            t_l2, t_l3 = get_feature_l2_l3(model, temp_t, mode='visible')
            s_l2, s_l3 = get_feature_l2_l3(model, ir_t, mode='thermal')
            
            # 3. 处理 Layer 2 (上采样 x2 -> Stride 4)
            t_l2 = F.interpolate(t_l2, scale_factor=2, mode='bilinear', align_corners=False)
            s_l2 = F.interpolate(s_l2, scale_factor=2, mode='bilinear', align_corners=False)
            t_l2 = F.normalize(t_l2, p=2, dim=1)
            s_l2 = F.normalize(s_l2, p=2, dim=1)
            
            # 4. 处理 Layer 3 (上采样 x4 -> Stride 4，与 Layer 2 对齐)
            # 注意：这里需要把 L3 放大到和 L2 一样大，以便后续相乘
            t_l3 = F.interpolate(t_l3, scale_factor=4, mode='bilinear', align_corners=False)
            s_l3 = F.interpolate(s_l3, scale_factor=4, mode='bilinear', align_corners=False)
            t_l3 = F.normalize(t_l3, p=2, dim=1)
            s_l3 = F.normalize(s_l3, p=2, dim=1)

            if t_l2.shape[2] > s_l2.shape[2] or t_l2.shape[3] > s_l2.shape[3]:
                continue

            # 5. 分别计算相关性得分图 (Score Maps)
            res_l2 = F.conv2d(s_l2, t_l2) # [1, 1, H, W]
            res_l3 = F.conv2d(s_l3, t_l3) # [1, 1, H, W]
            
            # 裁剪尺寸对齐 (防止 conv2d 后尺寸微小差异)
            min_h = min(res_l2.shape[2], res_l3.shape[2])
            min_w = min(res_l2.shape[3], res_l3.shape[3])
            res_l2 = res_l2[:, :, :min_h, :min_w]
            res_l3 = res_l3[:, :, :min_h, :min_w]

            # 6. --- 核心：分数融合 (Ensemble) ---
            # 归一化到 0~1 之间方便相乘 (ReLU 滤除负相关)
            score_l2 = F.relu(res_l2)
            score_l3 = F.relu(res_l3)
            
            # 乘法融合：既要位置对(L2)，又要语义对(L3)
            # 相当于逻辑 AND 操作
            final_score = score_l2 * score_l3 
            
            score_map = final_score.squeeze().cpu().numpy()
            _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
            
            stride = 4
            pred_x = int(max_loc[0] * stride)
            pred_y = int(max_loc[1] * stride)
            
            pred_box = [pred_x, pred_y, tw, th]
            iou = calculate_iou(item['bbox'], pred_box)
            all_ious.append(iou)
            
            if (i + 1) % 10 == 0:
                res_img = cv2.imread(item['ir_path'])
                cv2.rectangle(res_img, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)
                cv2.rectangle(res_img, (pred_x, pred_y), (pred_x+tw, pred_y+th), (0, 0, 255), 2)
                cv2.putText(res_img, f"IoU:{iou:.2f}", (pred_x, pred_y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(f"{RESULT_DIR}/{item['file_id']}.jpg", res_img)
                
        except Exception:
            continue

    if len(all_ious) > 0:
        ious = np.array(all_ious)
        print(f"\n--- 双层评分融合实验报告 ---")
        print(f"mIoU: {np.mean(ious):.4f}")
        print(f"SR@0.5: {np.sum(ious>0.5)/len(ious):.2%}")
        print(f"SR@0.3: {np.sum(ious>0.3)/len(ious):.2%}")

if __name__ == '__main__':
    main()