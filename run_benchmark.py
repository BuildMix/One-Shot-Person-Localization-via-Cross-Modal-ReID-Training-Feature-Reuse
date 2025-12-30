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
import time  # <--- 新增时间模块

# ==================== 配置区域 ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 默认指向 ICCV21 文件夹下的 json，请确保文件存在
JSON_PATH = os.path.join(BASE_DIR, 'ICCV21_CAJ/llvip_test_pairs.json') 

# 实验配置
EXPERIMENTS = [
    {
        "name": "ICCV21_CAJ_lr001",
        "code_dir": os.path.join(BASE_DIR, "ICCV21_CAJ"),
        "weight_path": "save_model/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.01_seed_0_best.t"
    },
    {
        "name": "ICCV21_CAJ_lr01",
        "code_dir": os.path.join(BASE_DIR, "ICCV21_CAJ"),
        "weight_path": "save_model/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.1_seed_0_best.t"
    },
    {
        "name": "PAMI23_Supervised",
        "code_dir": os.path.join(BASE_DIR, "PAMI23_Supervised"),
        "weight_path": "save_model/sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.1_seed_0_best.t" 
    }
]

TEST_LAYERS = [1, 2, 3, 4]
TEST_UPSAMPLE = [False, True]
LOG_FILE = "result.log"

# ================================================

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

def load_dynamic_model(code_dir, weight_rel_path):
    """动态加载指定目录下的 model.py 和权重"""
    # ... (保持原有的加载逻辑不变) ...
    print(f"正在加载环境: {code_dir} ...")
    
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    
    if 'model' in sys.modules:
        del sys.modules['model']
    import model
    importlib.reload(model)
    
    try:
        net = model.embed_net(class_num=395, no_local='on', gm_pool='on', arch='resnet50')
    except Exception as e:
        print(f"模型初始化尝试默认参数: {e}")
        net = model.embed_net(class_num=395, arch='resnet50')

    weight_path = os.path.join(code_dir, weight_rel_path)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件未找到: {weight_path}")
        
    print(f"加载权重: {weight_path}")
    checkpoint = torch.load(weight_path)
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    net.load_state_dict(new_state_dict, strict=False)
    net.eval()
    sys.path.pop(0)
    return net.cuda() if torch.cuda.is_available() else net

def extract_feature(model, img, mode, layer_idx, use_saj=True):
    """提取指定层的特征"""
    with torch.no_grad():
        x = model.visible_module(img) if mode == 'visible' else model.thermal_module(img)
        x = model.base_resnet.base.layer1(x)
        if layer_idx == 1:
            pass
        elif layer_idx >= 2:
            x = model.base_resnet.base.layer2(x)
            if layer_idx >= 3:
                x = model.base_resnet.base.layer3(x)
                if layer_idx == 4:
                    x = model.base_resnet.base.layer4(x)
        
        # PAMI23 SAJ 应用
        if use_saj and hasattr(model, 'spatial_attention'):
            x = model.spatial_attention(x)
            
    return x

def log_message(msg):
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def run_test_round(model, data_pairs, layer, do_upsample, method_name):
    """运行单轮测试并计时"""
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_ious = []
    
    base_strides = {1: 4, 2: 8, 3: 16, 4: 32}
    current_stride = base_strides[layer]
    if do_upsample:
        current_stride = current_stride / 2.0
    
    desc = f"[{method_name}] L{layer} {'+Up' if do_upsample else 'Raw'}"
    
    # === 计时开始 ===
    start_time = time.time()
    
    for item in tqdm(data_pairs, desc=desc, leave=False):
        try:
            # 1. 数据加载与预处理 (这也算在处理时间内)
            vis_img = Image.open(item['vis_path']).convert('RGB')
            ir_img = Image.open(item['ir_path']).convert('RGB')
            gt_box = item['bbox']
            tx, ty, tw, th = gt_box
            
            template = vis_img.crop((tx, ty, tx + tw, ty + th))
            temp_t = common_transform(template).unsqueeze(0).cuda()
            ir_t = common_transform(ir_img).unsqueeze(0).cuda()
            
            # 2. 推理过程
            feat_temp = extract_feature(model, temp_t, 'visible', layer)
            feat_search = extract_feature(model, ir_t, 'thermal', layer)
            
            if do_upsample:
                feat_temp = F.interpolate(feat_temp, scale_factor=2, mode='bilinear', align_corners=False)
                feat_search = F.interpolate(feat_search, scale_factor=2, mode='bilinear', align_corners=False)
            
            feat_temp = F.normalize(feat_temp, p=2, dim=1)
            feat_search = F.normalize(feat_search, p=2, dim=1)
            
            if feat_temp.shape[2] > feat_search.shape[2] or feat_temp.shape[3] > feat_search.shape[3]:
                continue
                
            res = F.conv2d(feat_search, feat_temp)
            score_map = res.squeeze().cpu().numpy()
            _, _, _, max_loc = cv2.minMaxLoc(score_map)
            
            pred_x = int(max_loc[0] * current_stride)
            pred_y = int(max_loc[1] * current_stride)
            pred_box = [pred_x, pred_y, tw, th]
            
            iou = calculate_iou(gt_box, pred_box)
            all_ious.append(iou)
            
        except Exception:
            continue
    
    # === 计时结束 ===
    end_time = time.time()
    total_cost = end_time - start_time
    
    # 防止除以零
    count = len(data_pairs)
    if count == 0:
        return 0, 0, 0, 0, 0
        
    avg_cost_ms = (total_cost / count) * 1000 # 转换为毫秒
    
    ious = np.array(all_ious) if len(all_ious) > 0 else np.array([0])
    miou = np.mean(ious)
    sr50 = np.sum(ious > 0.5) / len(ious)
    sr30 = np.sum(ious > 0.3) / len(ious)
    
    # 返回增加了时间指标
    return miou, sr50, sr30, total_cost, avg_cost_ms

def main():
    # 初始化日志文件，增加时间列
    with open(LOG_FILE, 'w') as f:
        f.write("Method, Layer, Upsample, mIoU, SR@0.5, SR@0.3, TotalTime(s), AvgTime(ms/img)\n")
    
    if not os.path.exists(JSON_PATH):
        print(f"错误: 找不到数据集索引文件 {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        data_pairs = json.load(f)
        
    print(f"=== 开始全量基准测试 (含耗时统计) ===")
    print(f"测试样本数: {len(data_pairs)}")
    print(f"日志路径: {os.path.abspath(LOG_FILE)}\n")

    for exp in EXPERIMENTS:
        try:
            model = load_dynamic_model(exp['code_dir'], exp['weight_path'])
            has_saj = hasattr(model, 'spatial_attention')
            saj_status = "(含 SAJ)" if has_saj else "(无 SAJ)"
            log_message(f"\n>>> 载入模型: {exp['name']} {saj_status}")
            
            for layer in TEST_LAYERS:
                for upsample in TEST_UPSAMPLE:
                    # 获取这一轮的 5 个返回值
                    miou, sr50, sr30, t_total, t_avg = run_test_round(
                        model, data_pairs, layer, upsample, exp['name']
                    )
                    
                    up_str = "True" if upsample else "False"
                    
                    # 打印到屏幕 (精简版)
                    res_str = f"{exp['name']}, L{layer}, Up={up_str} | SR@0.5: {sr50:.2%} | Time: {t_avg:.1f}ms"
                    print(f"   -> {res_str}")
                    
                    # 写入日志 (详细版)
                    with open(LOG_FILE, 'a') as f:
                        f.write(f"{exp['name']}, {layer}, {upsample}, {miou:.4f}, {sr50:.4f}, {sr30:.4f}, {t_total:.2f}, {t_avg:.2f}\n")
                        
        except Exception as e:
            log_message(f"!!! 实验 {exp['name']} 运行出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n所有测试完成！请查看 {LOG_FILE}")
    os.system('/root/upload.sh')

if __name__ == '__main__':
    main()