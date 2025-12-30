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

def get_feature_layer2(model, img_tensor, mode='visible'):
    """提取 Layer 2 特征 (原始步长 8)"""
    with torch.no_grad():
        x = model.visible_module(img_tensor) if mode == 'visible' else model.thermal_module(img_tensor)
        x = model.base_resnet.base.layer1(x)
        x = model.base_resnet.base.layer2(x)
    return x

def main():
    # --- 配置区域 ---
    MODEL_PATH = 'save_model/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.01_seed_0_best.t' # 请替换为你的真实权重文件名
    JSON_PATH = 'llvip_test_pairs.json'
    RESULT_DIR = 'search_results_upsample_3'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 基础预处理
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_network(MODEL_PATH)
    with open(JSON_PATH, 'r') as f:
        data_pairs = json.load(f)

    all_ious = []
    print(f"开始特征上采样实验 (Upsample x2, Stride=4)...")
    
    for i, item in enumerate(tqdm(data_pairs)):
        try:
            # 1. 准备数据
            vis_img = Image.open(item['vis_path']).convert('RGB')
            ir_img = Image.open(item['ir_path']).convert('RGB')
            gt_box = item['bbox'] # [x, y, w, h]
            tx, ty, tw, th = gt_box
            
            template = vis_img.crop((tx, ty, tx + tw, ty + th))
            
            temp_t = common_transform(template).unsqueeze(0).cuda()
            ir_t = common_transform(ir_img).unsqueeze(0).cuda()
            
            # 2. 提取特征 (Layer 2)
            feat_temp = get_feature_layer2(model, temp_t, mode='visible')
            feat_search = get_feature_layer2(model, ir_t, mode='thermal')
            
            # --- 关键改进：双线性插值上采样 2 倍 ---
            # align_corners=False 是目前的标准做法
            feat_temp = F.interpolate(feat_temp, scale_factor=2, mode='bilinear', align_corners=False)
            feat_search = F.interpolate(feat_search, scale_factor=2, mode='bilinear', align_corners=False)
            
            # 3. 归一化与匹配
            feat_temp = F.normalize(feat_temp, p=2, dim=1)
            feat_search = F.normalize(feat_search, p=2, dim=1)
            
            # 尺寸校验
            if feat_temp.shape[2] > feat_search.shape[2] or feat_temp.shape[3] > feat_search.shape[3]:
                continue

            res = F.conv2d(feat_search, feat_temp)
            
            # 4. 获取坐标
            score_map = res.squeeze().cpu().numpy()
            _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
            
            # --- 关键参数调整 ---
            # 原始 Stride 是 8，上采样 2 倍后，有效 Stride 变为 4
            stride = 4 
            
            pred_x = int(max_loc[0] * stride)
            pred_y = int(max_loc[1] * stride)
            
            pred_box = [pred_x, pred_y, tw, th]
            
            # 5. 评价
            iou = calculate_iou(gt_box, pred_box)
            all_ious.append(iou)
            
            # 6. 每 10 张保存
            if (i + 1) % 10 == 0:
                res_img = cv2.imread(item['ir_path'])
                cv2.rectangle(res_img, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)
                cv2.rectangle(res_img, (pred_x, pred_y), (pred_x+tw, pred_y+th), (0, 0, 255), 2)
                cv2.putText(res_img, f"IoU:{iou:.2f}", (pred_x, pred_y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(f"{RESULT_DIR}/{item['file_id']}.jpg", res_img)
                
        except Exception as e:
            continue

    if len(all_ious) > 0:
        ious = np.array(all_ious)
        print(f"\n--- 上采样优化报告 ---")
        print(f"mIoU: {np.mean(ious):.4f}")
        print(f"SR@0.5: {np.sum(ious>0.5)/len(ious):.2%}")
        print(f"SR@0.3: {np.sum(ious>0.3)/len(ious):.2%}")
        print(f"说明: 步长降低为4像素，定位理论精度提升一倍。")

if __name__ == '__main__':
    main()