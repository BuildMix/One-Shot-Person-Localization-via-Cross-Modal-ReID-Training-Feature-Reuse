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

def get_feature_v3(model, img_tensor, mode='visible'):
    """使用 Layer 2 特征，对空间位置和尺度更敏感"""
    with torch.no_grad():
        x = model.visible_module(img_tensor) if mode == 'visible' else model.thermal_module(img_tensor)
        x = model.base_resnet.base.layer1(x)
        x = model.base_resnet.base.layer2(x)
        # 停止在 Layer 2，总步长为 8 (visible_module 4x * layer2 2x)
    return x

def main():
    MODEL_PATH = 'save_model/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.01_seed_0_best.t' # 修改为你的文件名
    JSON_PATH = 'llvip_test_pairs.json'
    RESULT_DIR = 'search_results_v3'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 统一预处理：不使用 Resize，只做标准化
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_network(MODEL_PATH)
    with open(JSON_PATH, 'r') as f:
        data_pairs = json.load(f)

    all_ious = []
    print("开始尺度对齐实验 (v3)...")
    
    for i, item in enumerate(tqdm(data_pairs)):
        # 1. 裁剪模板 (保持原始尺寸！)
        vis_img = Image.open(item['vis_path']).convert('RGB')
        tx, ty, tw, th = item['bbox']
        template = vis_img.crop((tx, ty, tx + tw, ty + th))
        
        # 2. 转换张量
        temp_t = common_transform(template).unsqueeze(0).cuda()
        ir_img = Image.open(item['ir_path']).convert('RGB')
        ir_t = common_transform(ir_img).unsqueeze(0).cuda()
        
        # 3. 提取特征
        feat_temp = get_feature_v3(model, temp_t, mode='visible')
        feat_search = get_feature_v3(model, ir_t, mode='thermal')
        
        # 4. 匹配 (Cosine Similarity Convolution)
        feat_temp = F.normalize(feat_temp, p=2, dim=1)
        feat_search = F.normalize(feat_search, p=2, dim=1)
        
        # 检查核大小是否超过全图特征图
        if feat_temp.shape[2] > feat_search.shape[2] or feat_temp.shape[3] > feat_search.shape[3]:
            continue

        res = F.conv2d(feat_search, feat_temp)
        
        # 5. 寻找最大值并映射
        score_map = res.squeeze().cpu().numpy()
        _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
        
        # Layer 2 的有效步长是 8
        stride = 8
        pred_x = int(max_loc[0] * stride)
        pred_y = int(max_loc[1] * stride)
        
        # 构造预测框 (大小与原始模板一致)
        pred_box = [pred_x, pred_y, tw, th]
        
        # 6. 评价与保存
        iou = calculate_iou(item['bbox'], pred_box)
        all_ious.append(iou)
        
        if (i + 1) % 10 == 0:
            res_img = cv2.imread(item['ir_path'])
            cv2.rectangle(res_img, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2) # 绿框：GT
            cv2.rectangle(res_img, (pred_x, pred_y), (pred_x+tw, pred_y+th), (0, 0, 255), 2) # 红框：Pred
            cv2.imwrite(f"{RESULT_DIR}/{item['file_id']}.jpg", res_img)

    ious = np.array(all_ious)
    print(f"\n报告: mIoU: {np.mean(ious):.4f} | SR@0.5: {np.sum(ious>0.5)/len(ious):.2%}")

if __name__ == '__main__':
    main()