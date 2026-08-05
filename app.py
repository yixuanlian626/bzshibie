from ultralytics import YOLO
import cv2
import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import re
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys

# ========== 清屏（可选） ==========
# os.system('cls' if os.name == 'nt' else 'clear')

# ========== 【新增】打包后的路径处理函数 ==========
def resource_path(relative_path):
    """获取打包后资源的绝对路径"""
    if hasattr(sys, '_MEIPASS'):
        # 打包后运行，资源在临时解压目录
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)

def find_model():
    """自动查找 best.pt"""
    # 1. 先检查当前目录
    if os.path.exists('best.pt'):
        return 'best.pt'
    # 2. 检查打包后的临时目录
    if hasattr(sys, '_MEIPASS'):
        temp_path = os.path.join(sys._MEIPASS, 'best.pt')
        if os.path.exists(temp_path):
            return temp_path
    # 3. 找不到则返回None
    return None

print("=" * 60)
print("        YOLOv8 数码管数字批量识别工具")
print("=" * 60)

# ========== 1. 输入模型路径 ==========
print("\n【1/5】请指定模型文件路径")
print("-" * 50)

# 自动查找模型
auto_model = find_model()
if auto_model:
    print(f"✅ 自动找到模型: {auto_model}")
    use_auto = input("是否使用该模型？(y/n，默认 y): ").strip().lower()
    if use_auto != 'n':
        model_path = auto_model
    else:
        # 手动输入
        while True:
            model_path = input("🤖 模型文件路径: ").strip().strip('"').strip("'")
            if os.path.exists(model_path):
                break
            else:
                print("   ❌ 文件不存在，请重新输入")
else:
    while True:
        model_path = input("🤖 模型文件路径: ").strip().strip('"').strip("'")
        if os.path.exists(model_path):
            break
        else:
            print("   ❌ 文件不存在，请重新输入")

# 加载模型
print(f"⏳ 正在加载模型...")
model = YOLO(model_path)
print(f"✅ 模型加载成功: {model_path}")

# ========== 2. 输入图片文件夹路径 ==========
print("\n【2/5】请指定待识别的图片文件夹")
print("-" * 50)
while True:
    image_folder = input("📁 图片文件夹路径: ").strip().strip('"').strip("'")
    if os.path.exists(image_folder):
        break
    else:
        print("   ❌ 路径不存在，请重新输入")

# ========== 3. 输入CSV输出路径 ==========
print("\n【3/5】请指定CSV结果保存位置")
print("-" * 50)
csv_path = input("📊 CSV输出路径（如 D:/results.csv 或 D:/folder/）: ").strip().strip('"').strip("'")

# 如果输入的是文件夹路径，自动补全文件名
if os.path.isdir(csv_path):
    csv_path = os.path.join(csv_path, 'recognition_results.csv')
elif not csv_path.endswith('.csv'):
    csv_path = csv_path + '.csv'

# 确保目录存在
csv_dir = os.path.dirname(csv_path)
if csv_dir:
    os.makedirs(csv_dir, exist_ok=True)

# ========== 4. 输入结果图片保存路径 ==========
print("\n【4/5】请指定结果图片保存位置")
print("-" * 50)
result_folder = input("🖼️ 结果图片保存路径（如 D:/result_images/，留空则保存在CSV同目录）: ").strip().strip('"').strip("'")

if not result_folder:
    result_folder = os.path.join(os.path.dirname(csv_path), 'result_images')
os.makedirs(result_folder, exist_ok=True)

# ========== 5. 是否保存结果图片 ==========
save_images_input = input("💾 是否保存带框的结果图片？(y/n，默认 y): ").strip().lower()
save_result_images = save_images_input != 'n'

# ========== 显示配置确认 ==========
print("\n" + "=" * 60)
print("📋 配置确认")
print("=" * 60)
print(f"🤖 模型文件:   {model_path}")
print(f"📁 图片文件夹: {image_folder}")
print(f"📊 CSV输出:    {csv_path}")
print(f"🖼️ 结果图片:   {result_folder}")
print(f"💾 保存图片:   {'是' if save_result_images else '否'}")
print("=" * 60)

confirm = input("\n确认以上配置？(y/n，默认 y): ").strip().lower()
if confirm == 'n':
    print("已取消运行")
    exit()

# ========== 开始处理 ==========
print("\n⏳ 开始处理...\n")

# ========== 获取所有图片（按文件名排序） ==========
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_files = []

# 使用rglob和字符串匹配，不区分大小写
for ext in image_extensions:
    # 使用glob的字符串方法，让glob自己处理大小写
    # 方法：遍历文件夹，检查文件扩展名（不区分大小写）
    for file_path in Path(image_folder).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

# 去重并排序
image_files = sorted(set(image_files), key=lambda x: x.name)

# 按文件名排序（确保frame_0000, frame_0001, ... 的顺序）
image_files = sorted(image_files, key=lambda x: x.name)

print(f"📁 找到 {len(image_files)} 张图片\n")

if len(image_files) == 0:
    print("❌ 未找到任何图片，请检查文件夹路径")
    exit()

# ========== 提取帧序号的函数 ==========
def extract_frame_number(filename):
    """从文件名中提取帧序号"""
    # 匹配各种可能的帧命名格式
    patterns = [
        r'(\d+)',  # 纯数字
        r'frame[_\s-]?(\d+)',  # frame_0000, frame0000, frame-0000
        r'img[_\s-]?(\d+)',  # img_0000, img0000
        r'pic[_\s-]?(\d+)',  # pic_0000, pic0000
        r'f(\d+)',  # f0000
        r'(\d{4})',  # 四位数字
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # 如果无法提取，返回None
    return None

# ========== 批量处理 ==========
results_data = []
csv_headers = ['时间(s)', '电动势', '置信度']

for idx, img_path in enumerate(image_files):
    print(f"🔄 正在处理 [{idx+1}/{len(image_files)}]: {img_path.name}")
    
    # 预测
    results = model(str(img_path))
    boxes = results[0].boxes
    
    detected = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x_center = float(box.xywh[0][0])
            detected.append((x_center, cls, conf))
    
    if len(detected) == 0:
        full_number = ''
        avg_conf = 0
        print(f"   ⚠️ 未检测到数字")
    else:
        detected.sort(key=lambda x: x[0])
        digits = [str(d[1]) for d in detected]
        confidences = [d[2] for d in detected]
        full_number = ''.join(digits)
        avg_conf = sum(confidences) / len(confidences)
        print(f"   ✅ 识别结果: {full_number} (置信度: {avg_conf:.3f})")
    
    # 提取帧序号作为时间
    frame_num = extract_frame_number(img_path.name)
    if frame_num is not None:
        time_sec = frame_num  # 直接使用帧序号作为时间（单位：秒）
    else:
        # 如果无法提取帧序号，使用索引
        time_sec = idx
        print(f"   ⚠️ 无法从文件名提取帧序号，使用索引: {time_sec}")
    
    # 保存数据：时间(s)、电动势、置信度
    results_data.append([
        time_sec,
        full_number if full_number else 'N/A',
        f"{avg_conf:.3f}" if avg_conf > 0 else '0.000'
    ])
    
    # 保存带框结果图片
    if save_result_images and boxes is not None and len(detected) > 0:
        img = results[0].orig_img.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f'{cls} {conf:.2f}'
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y2), (x1 + text_w + 4, y2 + text_h + 8), (0, 0, 0), -1)
            cv2.putText(img, label, (x1 + 2, y2 + text_h + 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(img, f'Result: {full_number}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        result_path = os.path.join(result_folder, f'result_{time_sec:04d}_{img_path.name}')
        cv2.imwrite(result_path, img)

# ========== 按时间排序 ==========
results_data.sort(key=lambda x: x[0])

# ========== 生成CSV（只有三列） ==========
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(results_data)

# ========== 完成识别 ==========
print("\n" + "=" * 60)
print("✅ 识别完成！")
print("=" * 60)
print(f"📊 CSV结果已保存: {csv_path}")
if save_result_images:
    print(f"🖼️ 结果图片已保存: {result_folder}")
print(f"📁 共处理: {len(image_files)} 张图片")
print("=" * 60)

# 预览前10条
print("\n📋 结果预览（前10条）:")
print("-" * 70)
print(f"  {'时间(s)':<10} | {'电动势':<15} | {'置信度':<10}")
print("-" * 70)
for row in results_data[:10]:
    print(f"  {row[0]:<10} | {row[1]:<15} | {row[2]:<10}")
if len(results_data) > 10:
    print(f"  ... 共 {len(results_data)} 条记录")

# ========== 绘制电动势-时间平滑曲线 ==========
print("\n" + "=" * 60)
print("📈 开始生成电动势-时间平滑曲线...")
print("=" * 60)

# 询问是否生成曲线图
plot_choice = input("\n是否生成电动势-时间平滑曲线图？(y/n，默认 y): ").strip().lower()
if plot_choice != 'n':
    try:
        # 从results_data中提取数据
        times = []
        emfs = []
        confidences = []
        
        for row in results_data:
            time_sec = row[0]
            emf_str = row[1]
            conf_str = row[2]
            
            # 跳过无效数据
            if emf_str == 'N/A' or emf_str == '':
                continue
            
            try:
                emf_value = float(emf_str)
                conf_value = float(conf_str)
            except:
                continue
            
            times.append(float(time_sec))
            emfs.append(emf_value)
            confidences.append(conf_value)
        
        if len(times) == 0 or len(emfs) == 0:
            print("❌ 没有有效数据生成曲线")
        else:
            # 转换为numpy数组
            times = np.array(times)
            emfs = np.array(emfs)
            confidences = np.array(confidences)
            
            # 按时间排序（已经排过序，但以防万一）
            sort_idx = np.argsort(times)
            times_sorted = times[sort_idx]
            emfs_sorted = emfs[sort_idx]
            confidences_sorted = confidences[sort_idx]

            # ========== 【新增】过滤异常值（电动势 > 1000 和< 100 视为异常） ==========
            filter_mask = (emfs_sorted >= 100) & (emfs_sorted <= 1000)
            times_filtered = times_sorted[filter_mask]
            emfs_filtered = emfs_sorted[filter_mask]
            confidences_filtered = confidences_sorted[filter_mask]
            print(f"   📊 过滤前: {len(times_sorted)} 个点，过滤后: {len(times_filtered)} 个点（已过滤异常值 > 1000 和 < 100 ）")

            # 使用过滤后的数据
            times_plot = times_filtered
            emfs_plot = emfs_filtered
            confidences_plot = confidences_filtered
            
            # 计算数据的范围，用于确定合适的分度值
            time_range = times_plot.max() - times_plot.min()
            emf_range = emfs_plot.max() - emfs_plot.min()

            # ========== 【修改】手动指定刻度间隔 ==========
            x_interval = 50   # 横坐标每 50 秒一个刻度
            y_interval = 50   # 纵坐标每 50 mV 一个刻度
            
            # ========== 【修改】增大图形尺寸 ==========
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
            
            # ========== 图1: 原始数据点（用置信度作为颜色） ==========
            # ========== 【修改】点的大小从80改为10 ==========
            scatter = ax1.scatter(times_plot, emfs_plot, 
                                 c=confidences_plot, cmap='viridis', 
                                 s=10, alpha=0.6, zorder=5)
            ax1.plot(times_plot, emfs_plot, 'b--', alpha=0.3, label='Connection Line')
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('EMF (mV)', fontsize=12)
            ax1.set_title('EMF vs Time - Raw Data (All Points)', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 设置坐标轴刻度
            ax1.xaxis.set_major_locator(MultipleLocator(x_interval))
            ax1.yaxis.set_major_locator(MultipleLocator(y_interval))
            # 设置刻度标签格式
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            # 自动调整刻度标签密度
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.colorbar(scatter, ax=ax1, label='Confidence')
            
            # ========== 图2: 平滑曲线 ==========
            if len(times_plot) >= 4:  # 至少需要4个点才能做样条插值
                try:
                    # 生成平滑曲线
                    n_points = 300
                    x_smooth = np.linspace(times_plot.min(), times_plot.max(), n_points)
                    
                    # 使用B样条插值
                    spl = make_interp_spline(times_plot, emfs_plot, k=min(3, len(times_plot)-1))
                    y_smooth = spl(x_smooth)
                    
                    ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smooth Curve')
                    # ========== 【修改】点的大小从50改为10 ==========
                    ax2.scatter(times_plot, emfs_plot, color='blue', s=10, label='Raw Data', zorder=5)
                    ax2.set_xlabel('Time (s)', fontsize=12)
                    ax2.set_ylabel('EMF (mV)', fontsize=12)
                    ax2.set_title('EMF vs Time - Smooth Curve', fontsize=14)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    # 设置坐标轴刻度
                    ax2.xaxis.set_major_locator(MultipleLocator(x_interval))
                    ax2.yaxis.set_major_locator(MultipleLocator(y_interval))
                    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    # 在平滑曲线上标记极值点
                    if len(y_smooth) > 10:
                        from scipy.signal import find_peaks
                        peaks, _ = find_peaks(y_smooth, height=(emfs_plot.min() + 0.1*(emfs_plot.max()-emfs_plot.min()), None))
                        valleys, _ = find_peaks(-y_smooth, height=(-(emfs_plot.max() - 0.1*(emfs_plot.max()-emfs_plot.min())), None))
                        
                        for peak_idx in peaks:
                            ax2.axvline(x_smooth[peak_idx], color='green', linestyle=':', alpha=0.5, label='Peak' if peak_idx == peaks[0] else '')
                        for valley_idx in valleys:
                            ax2.axvline(x_smooth[valley_idx], color='orange', linestyle=':', alpha=0.5, label='Valley' if valley_idx == valleys[0] else '')
                        
                        if len(peaks) > 0 or len(valleys) > 0:
                            ax2.legend()
                    
                except Exception as e:
                    print(f"   ⚠️ 平滑插值失败: {e}，改用线性插值")
                    # 使用线性插值作为备选
                    x_smooth = np.linspace(times_plot.min(), times_plot.max(), 300)
                    y_smooth = np.interp(x_smooth, times_plot, emfs_plot)
                    ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smooth Curve')
                    ax2.scatter(times_plot, emfs_plot, color='blue', s=10, label='Raw Data', zorder=5)
                    ax2.set_xlabel('Time (s)', fontsize=12)
                    ax2.set_ylabel('EMF (mV)', fontsize=12)
                    ax2.set_title('EMF vs Time - Smooth Curve', fontsize=14)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    # 设置坐标轴刻度
                    ax2.xaxis.set_major_locator(MultipleLocator(x_interval))
                    ax2.yaxis.set_major_locator(MultipleLocator(y_interval))
                    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                # 点数太少，只绘制连线
                ax2.plot(times_plot, emfs_plot, 'r-o', linewidth=2, label='Data Connection Line')
                ax2.scatter(times_plot, emfs_plot, color='blue', s=10, label='Raw Data', zorder=5)
                ax2.set_xlabel('Time (s)', fontsize=12)
                ax2.set_ylabel('EMF (mV)', fontsize=12)
                ax2.set_title('EMF vs Time (Few points, displayed as lines)', fontsize=14)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 设置坐标轴刻度
                ax2.xaxis.set_major_locator(MultipleLocator(x_interval))
                ax2.yaxis.set_major_locator(MultipleLocator(y_interval))
                ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # 保存曲线图
            plot_save_path = os.path.join(os.path.dirname(csv_path), 'emf_time_curve.png')
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 曲线图已保存: {plot_save_path}")
            
            # 显示图形
            plt.show()
            
    except Exception as e:
        print(f"❌ 生成曲线图失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("🎉 所有任务完成！")
print("=" * 60)