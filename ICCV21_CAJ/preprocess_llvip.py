import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

def preprocess_llvip_for_detection(llvip_root):
    # 定义子目录路径
    anno_dir = os.path.join(llvip_root, 'Annotations')
    # 根据你的目录结构，test 图片位于 visible/test 和 infrared/test
    vis_test_dir = os.path.join(llvip_root, 'visible/test')
    #vis_test_dir = os.path.join(llvip_root, 'visible/train')
    
    if not os.path.exists(vis_test_dir):
        print(f"错误: 找不到目录 {vis_test_dir}。请检查 LLVIP_ROOT 路径是否正确。")
        return

    # 存储处理后的数据对
    test_pairs = []

    # 获取测试集所有图片名（不含后缀）
    test_files = [f.split('.')[0] for f in os.listdir(vis_test_dir) if f.endswith('.jpg')]
    print(f"找到 {len(test_files)} 张测试图片，开始解析标注...")

    for file_id in tqdm(test_files):
        xml_path = os.path.join(anno_dir, file_id + '.xml')
        
        if not os.path.exists(xml_path):
            continue

        # 解析 XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"解析 {xml_path} 出错: {e}")
            continue
        
        target_box = None
        
        # 寻找目标，LLVIP 官方标签通常为 'person'
        for obj in root.findall('object'):
            name = obj.find('name').text.lower() # 转小写增强兼容性
            if name in ['person', 'pedestrian']:
                bbox = obj.find('bndbox')
                # 提取左上角和右下角坐标
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # 计算 w, h
                w = xmax - xmin
                h = ymax - ymin
                
                # 稍微放宽过滤条件，确保能抓到目标
                if w > 10 and h > 20:
                    target_box = [xmin, ymin, w, h]
                    break  # 只取每张图的第一个人进行简化
        
        if target_box:
            test_pairs.append({
                'file_id': file_id,
                'vis_path': os.path.join(llvip_root, 'visible/test', file_id + '.jpg'),
                'ir_path': os.path.join(llvip_root, 'infrared/test', file_id + '.jpg'),
                'bbox': target_box  # [x, y, w, h]
            })

    # 保存索引文件
    output_file = 'llvip_test_pairs.json'
    #output_file = 'llvip_train_pairs.json'
    with open(output_file, 'w') as f:
        json.dump(test_pairs, f, indent=4)
    
    print(f"\n预处理完成！")
    print(f"共提取有效单目标图片对: {len(test_pairs)}")
    print(f"索引文件已保存至: {os.path.abspath(output_file)}")

if __name__ == '__main__':
    # 路径设置：如果你在 ICCV21_CAJ 运行，LLVIP 在项目隔壁
    # LLVIP_ROOT 应该是相对于当前执行位置的路径
    # 建议使用绝对路径以防万一
    #LLVIP_ROOT = '../LLVIP' 
    #preprocess_llvip_for_detection(LLVIP_ROOT)
    # 修改为绝对路径，这样生成的 JSON 文件在任何地方都能读到图
    LLVIP_ROOT = '/hy-tmp/LLVIP' 
    preprocess_llvip_for_detection(LLVIP_ROOT)