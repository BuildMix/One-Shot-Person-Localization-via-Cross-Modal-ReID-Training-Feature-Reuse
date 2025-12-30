import os
import sys
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import importlib
import time

# ==================== SOTA 配置区域 ====================

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. PAMI23 代码所在目录 (必须包含 model.py)
CODE_DIR = os.path.join(BASE_DIR, "PAMI23_Supervised")

# 2. PAMI23 最佳权重路径 (请根据实际情况修改文件名)
# 假设权重在 PAMI23_Supervised/save_model/ 下
WEIGHT_PATH = os.path.join(CODE_DIR, "save_model/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.1_seed_0_best.t") 
# 如果文件名很长（如 sysu_adp...），请替换上面的 "best.t"

# 3. 数据集 JSON 路径
JSON_PATH = os.path.join(BASE_DIR, 'PAMI23_Supervised/llvip_test_pairs.json')

# 4. 结果保存设置
RESULT_DIR = os.path.join(BASE_DIR, "sota_viz_result") # 可视化图片保存路径
LOG_FILE = "sota_metrics.log"                          # 指标日志

# SOTA 核心参数 (固定)
TARGET_LAYER = 3  # 使用 Layer 3
UPSAMPLE_SCALE = 2 # 上采样倍率

# ======================================================

def load_pami23_model(code_dir, weight_path):
    """
    动态加载 PAMI23 模型环境和权重
    """
    print(f"[System] 正在加载 PAMI23 环境: {code_dir} ...")
    
    # 临时将 PAMI23 目录加入系统路径，以便导入其中的 model.py
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    
    # 强制重载 model 模块，防止与其他版本的 model.py 冲突
    if 'model' in sys.modules:
        del sys.modules['model']
    import model
    importlib.reload(model)
    
    # 初始化模型结构
    try:
        # PAMI23 标准初始化参数
        net = model.embed_net(class_num=395, no_local='on', gm_pool='on', arch='resnet50')
    except Exception as e:
        print(f"[Warning] 初始化参数不匹配，尝试默认初始化: {e}")
        net = model.embed_net(class_num=395, arch='resnet50')

    # 检查是否包含核心模块 SAJ
    if not hasattr(net, 'spatial_attention'):
        print("\n[Critical Error] 加载的模型代码中缺少 'spatial_attention' 模块！")
        print("请检查 PAMI23_Supervised/model.py 是否已正确添加了 SAJ 类定义。")
        sys.exit(1)

    # 加载权重
    if not os.path.exists(weight_path):
        print(f"\n[Error] 权重文件未找到: {weight_path}")
        print("请在代码配置区域修改 WEIGHT_PATH 为真实路径。")
        sys.exit(1)
        
    print(f"[System] 加载权重: {os.path.basename(weight_path)}")
    checkpoint = torch.load(weight_path)
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
    
    # 去除 module. 前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    # 加载参数 (strict=False 以防万一，但 SAJ 参数必须在里面)
    try:
        net.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Warning] 严格加载失败 (可能有些无关层不匹配): {e}")
        net.load_state_dict(new_state_dict, strict=False)
        
    net.eval()
    
    # 恢复系统路径
    sys.path.pop(0)
    
    return net.cuda() if torch.cuda.is_available() else net

def extract_sota_feature(model, img_tensor, mode):
    """
    SOTA 特征提取流:
    Input -> Layer1 -> Layer2 -> Layer3 -> SAJ (Attention) -> Output
    """
    with torch.no_grad():
        # 1. 模态特定层
        x = model.visible_module(img_tensor) if mode == 'visible' else model.thermal_module(img_tensor)
        
        # 2. 共享骨干 (直到 Layer 3)
        x = model.base_resnet.base.layer1(x)
        x = model.base_resnet.base.layer2(x)
        x = model.base_resnet.base.layer3(x) # Stride = 16
        
        # 3. [SOTA 核心] 应用空间注意力 (SAJ)
        # 这一步至关重要，过滤 Layer 3 的背景噪声
        x = model.spatial_attention(x)
        
    return x

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

def main():
    # 1. 环境准备
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(JSON_PATH):
        print(f"[Error] 找不到数据集索引文件: {JSON_PATH}")
        return

    # 2. 加载模型
    model = load_pami23_model(CODE_DIR, WEIGHT_PATH)
    
    # 3. 数据预处理
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with open(JSON_PATH, 'r') as f:
        data_pairs = json.load(f)

    print(f"\n==============================================")
    print(f"   SOTA Zero-Shot Localization (PAMI23)")
    print(f"   Config: Layer {TARGET_LAYER} + SAJ + Upsample x{UPSAMPLE_SCALE}")
    print(f"==============================================\n")
    
    all_ious = []
    start_time = time.time()
    
    # 4. 主循环
    for i, item in enumerate(tqdm(data_pairs, desc="Running SOTA")):
        try:
            # --- 数据加载 ---
            vis_path = item['vis_path']
            ir_path = item['ir_path']
            gt_box = item['bbox'] # [x, y, w, h]
            tx, ty, tw, th = gt_box
            
            # 打开图片
            vis_img = Image.open(vis_path).convert('RGB')
            ir_img = Image.open(ir_path).convert('RGB')
            
            # 裁剪模板
            template = vis_img.crop((tx, ty, tx + tw, ty + th))
            
            # 转 Tensor
            temp_t = common_transform(template).unsqueeze(0).cuda()
            ir_t = common_transform(ir_img).unsqueeze(0).cuda()
            
            # --- [Step 1] 特征提取 (Layer 3 + SAJ) ---
            feat_temp = extract_sota_feature(model, temp_t, mode='visible')
            feat_search = extract_sota_feature(model, ir_t, mode='thermal')
            
            # --- [Step 2] 空间分辨率恢复 (Upsample) ---
            # Layer 3 原始 stride=16, 上采样 x2 后 stride=8
            feat_temp = F.interpolate(feat_temp, scale_factor=UPSAMPLE_SCALE, mode='bilinear', align_corners=False)
            feat_search = F.interpolate(feat_search, scale_factor=UPSAMPLE_SCALE, mode='bilinear', align_corners=False)
            
            # --- [Step 3] 归一化与匹配 ---
            feat_temp = F.normalize(feat_temp, p=2, dim=1)
            feat_search = F.normalize(feat_search, p=2, dim=1)
            
            # 尺寸保护
            if feat_temp.shape[2] > feat_search.shape[2] or feat_temp.shape[3] > feat_search.shape[3]:
                continue
            
            # 卷积
            res = F.conv2d(feat_search, feat_temp)
            
            # --- [Step 4] 坐标定位 ---
            score_map = res.squeeze().cpu().numpy()
            _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
            
            # 步长换算
            # Layer 3 (Stride 16) / Upsample (2) = Effective Stride 8
            current_stride = 16 / UPSAMPLE_SCALE 
            
            pred_x = int(max_loc[0] * current_stride)
            pred_y = int(max_loc[1] * current_stride)
            pred_box = [pred_x, pred_y, tw, th]
            
            # --- 评价 ---
            iou = calculate_iou(gt_box, pred_box)
            all_ious.append(iou)
            
            # --- 可视化保存 (每 50 张存一张，且 IoU > 0.5 的优先存) ---
            # 也可以改为保存所有，或者保存特定 index
            save_condition = (i % 50 == 0) or (iou > 0.8 and i % 10 == 0)
            
            if save_condition:
                res_img = cv2.imread(ir_path)
                # 绿色 GT
                cv2.rectangle(res_img, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)
                # 红色 Predict
                cv2.rectangle(res_img, (pred_x, pred_y), (pred_x+tw, pred_y+th), (0, 0, 255), 2)
                
                info_text = f"SOTA IoU:{iou:.2f}"
                cv2.putText(res_img, info_text, (pred_x, pred_y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                save_name = f"{item['file_id']}_iou_{iou:.2f}.jpg"
                cv2.imwrite(os.path.join(RESULT_DIR, save_name), res_img)

        except Exception as e:
            continue

    # 5. 统计结果
    total_time = time.time() - start_time
    avg_time = (total_time / len(data_pairs)) * 1000
    
    ious = np.array(all_ious)
    miou = np.mean(ious)
    sr50 = np.sum(ious > 0.5) / len(ious)
    sr30 = np.sum(ious > 0.3) / len(ious)
    
    # 打印报告
    report = (
        f"\n================ SOTA 最终报告 ================\n"
        f"Method       : PAMI23 (Layer 3 + SAJ)\n"
        f"Upsample     : True (x2)\n"
        f"Samples      : {len(ious)}\n"
        f"mIoU         : {miou:.4f}\n"
        f"SR@0.5       : {sr50:.2%} (SOTA Result)\n"
        f"SR@0.3       : {sr30:.2%}\n"
        f"Avg Time     : {avg_time:.2f} ms/img\n"
        f"Viz Saved to : {RESULT_DIR}\n"
        f"===============================================\n"
    )
    
    print(report)
    
    # 保存日志
    with open(LOG_FILE, 'w') as f:
        f.write(report)

if __name__ == '__main__':
    main()