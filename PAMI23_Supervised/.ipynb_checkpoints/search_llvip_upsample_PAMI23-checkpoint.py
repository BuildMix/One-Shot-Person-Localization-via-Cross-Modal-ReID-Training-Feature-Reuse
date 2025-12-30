import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import sys

# 尝试导入 embed_net，确保 model.py 在当前目录下
try:
    from model import embed_net
except ImportError:
    print("错误: 找不到 model.py。请确保 search_llvip_pami23.py 和 model.py 在同一个文件夹下。")
    sys.exit(1)

# ================= 配置区域 (Config) =================
# 请修改为你实际训练好的 PAMI23 权重路径
MODEL_PATH = 'save_model/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.1_seed_0_best.t' 

# 数据集配置文件路径
JSON_PATH = 'llvip_test_pairs.json'

# 结果保存路径
RESULT_DIR = 'search_results_pami23'
# ====================================================

def calculate_iou(box1, box2):
    """计算两个框的 IoU (Intersection over Union)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

def load_network(checkpoint_path):
    """加载 PAMI23 模型权重"""
    print(f"正在加载模型: {checkpoint_path}")
    
    # 1. 初始化模型架构 (确保 arch='resnet50' 与训练时一致)
    model = embed_net(class_num=395, no_local='on', gm_pool='on', arch='resnet50')
    
    # 2. 加载权重文件
    if not os.path.exists(checkpoint_path):
        print(f"错误: 权重文件不存在 -> {checkpoint_path}")
        sys.exit(1)
        
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
    
    # 3. 移除 'module.' 前缀 (兼容多卡训练权重)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    # 4. 加载参数 (strict=False 以防万一有些不重要的层不匹配，但通常应该为 True)
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"警告: 权重加载主要部分匹配，但存在部分不匹配 (可能是 SAJ 模块缺失?): \n{e}")
        # 如果你确定 model.py 没改对，这里会报错
        
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model

def get_feature_pami23(model, img_tensor, mode='visible'):
    """
    PAMI23 核心特征提取函数
    流程: Input -> Layer1 -> Layer2 -> Spatial Attention -> Output
    """
    with torch.no_grad():
        # 1. 模态特定分支 (Visible / Thermal Module)
        if mode == 'visible':
            x = model.visible_module(img_tensor)
        else:
            x = model.thermal_module(img_tensor)
            
        # 2. 共享骨干提取 (Layer 1 & Layer 2)
        x = model.base_resnet.base.layer1(x)
        x = model.base_resnet.base.layer2(x) # Stride = 8
        
        # 3. [PAMI23 核心] 应用空间注意力 (SAJ)
        # 即使 SAJ 是在深层训练的，它的卷积核 (2->1) 学习的是"前景 vs 背景"的二值关系
        # 我们可以将其迁移到 Layer 2 使用，抑制 Layer 2 中的背景纹理
        if hasattr(model, 'spatial_attention'):
            x = model.spatial_attention(x)
        else:
            print("警告: 模型中未找到 spatial_attention 模块，退化为 ICCV21 模式")
            
    return x

def main():
    # 1. 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 2. 定义预处理 (ImageNet 标准化)
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 加载模型
    model = load_network(MODEL_PATH)

    # 4. 加载测试对列表
    if not os.path.exists(JSON_PATH):
        print(f"错误: 找不到测试列表文件 -> {JSON_PATH}")
        sys.exit(1)
        
    with open(JSON_PATH, 'r') as f:
        data_pairs = json.load(f)

    all_ious = []
    print(f"开始 PAMI23 检测实验 (Layer 2 + SAJ + Upsample x2)...")
    print(f"测试样本数: {len(data_pairs)}")
    
    # 5. 主循环
    for i, item in enumerate(tqdm(data_pairs)):
        try:
            # --- 数据读取 ---
            vis_path = item['vis_path']
            ir_path = item['ir_path']
            gt_box = item['bbox'] # [x, y, w, h]
            tx, ty, tw, th = gt_box
            
            # 打开图片
            vis_img = Image.open(vis_path).convert('RGB')
            ir_img = Image.open(ir_path).convert('RGB')
            
            # 裁剪模板 (可见光人像)
            template = vis_img.crop((tx, ty, tx + tw, ty + th))
            
            # 转 Tensor
            temp_t = common_transform(template).unsqueeze(0).cuda()
            ir_t = common_transform(ir_img).unsqueeze(0).cuda()
            
            # --- 特征提取 (PAMI23 模式) ---
            # 这一步会自动应用训练好的空间注意力来过滤背景
            feat_temp = get_feature_pami23(model, temp_t, mode='visible')
            feat_search = get_feature_pami23(model, ir_t, mode='thermal')
            
            # --- 空间分辨率恢复 (Upsample x2) ---
            # 将 Stride 从 8 降低到 4
            feat_temp = F.interpolate(feat_temp, scale_factor=2, mode='bilinear', align_corners=False)
            feat_search = F.interpolate(feat_search, scale_factor=2, mode='bilinear', align_corners=False)
            
            # --- 归一化 ---
            feat_temp = F.normalize(feat_temp, p=2, dim=1)
            feat_search = F.normalize(feat_search, p=2, dim=1)
            
            # 尺寸保护 (防止模板比搜索图还大)
            if feat_temp.shape[2] > feat_search.shape[2] or feat_temp.shape[3] > feat_search.shape[3]:
                continue

            # --- 卷积匹配 ---
            # feat_temp 作为卷积核
            res = F.conv2d(feat_search, feat_temp)
            
            # --- 坐标定位 ---
            score_map = res.squeeze().cpu().numpy()
            _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
            
            # 坐标映射 (Effective Stride = 4)
            stride = 4 
            pred_x = int(max_loc[0] * stride)
            pred_y = int(max_loc[1] * stride)
            
            # 构造预测框
            pred_box = [pred_x, pred_y, tw, th]
            
            # --- 评价 ---
            iou = calculate_iou(gt_box, pred_box)
            all_ious.append(iou)
            
            # --- 可视化 (每 20 张存一张) ---
            if (i + 1) % 20 == 0:
                res_img = cv2.imread(ir_path)
                # 绿色: 真实框
                cv2.rectangle(res_img, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)
                # 红色: 预测框 (PAMI23)
                cv2.rectangle(res_img, (pred_x, pred_y), (pred_x+tw, pred_y+th), (0, 0, 255), 2)
                
                info_text = f"PAMI23 IoU:{iou:.2f}"
                cv2.putText(res_img, info_text, (pred_x, pred_y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                save_path = os.path.join(RESULT_DIR, f"{item['file_id']}.jpg")
                cv2.imwrite(save_path, res_img)

        except Exception as e:
            # print(f"Error skipping: {e}")
            continue

    # 6. 输出最终报告
    if len(all_ious) > 0:
        ious = np.array(all_ious)
        print(f"\n========================================")
        print(f"      PAMI23 (L2+SAJ+Up) 检测报告       ")
        print(f"========================================")
        print(f"测试样本总数 : {len(ious)}")
        print(f"平均 IoU     : {np.mean(ious):.4f}")
        print(f"SR@0.5       : {np.sum(ious>0.5)/len(ious):.2%} (目标)")
        print(f"SR@0.3       : {np.sum(ious>0.3)/len(ious):.2%}")
        print(f"========================================")
        print(f"可视化结果已保存至: {RESULT_DIR}")
    else:
        print("未生成任何有效预测，请检查数据路径或 JSON 文件。")

if __name__ == '__main__':
    main()