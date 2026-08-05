import streamlit as st
from ultralytics import YOLO
import cv2
import os
import csv
import io
import zipfile
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import re
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pathlib import Path
import pandas as pd

# ========== 1. 页面配置 ==========
st.set_page_config(page_title="数码管批量识别", layout="wide")
st.title("📟 数码管数字批量识别工具")
st.markdown("上传包含数码管图片的 **ZIP 压缩包**，系统将自动识别所有图片中的数字组合并生成 CSV 结果。")

# ========== 2. 加载模型（使用缓存） ==========
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ 模型文件 '{model_path}' 未找到，请确保它位于项目根目录。")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None

model = load_model()
if model is None:
    st.stop()  # 模型加载失败则停止

# ========== 3. 侧边栏：参数设置 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")
    # 文件上传
    uploaded_zip = st.file_uploader(
        "上传图片压缩包 (ZIP)",
        type=['zip'],
        help="请将图片打包成 ZIP 格式上传"
    )
    # 是否保存结果图片
    save_images = st.checkbox("保存带检测框的结果图片", value=True)
    # 是否生成曲线图
    generate_plot = st.checkbox("生成电动势-时间平滑曲线图", value=True)
    # 置信度阈值
    conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05)

# ========== 4. 核心处理函数 ==========
def extract_frame_number(filename):
    """从文件名中提取帧序号"""
    patterns = [
        r'(\d+)',
        r'frame[_\s-]?(\d+)',
        r'img[_\s-]?(\d+)',
        r'pic[_\s-]?(\d+)',
        r'f(\d+)',
        r'(\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def process_images(image_files, model, save_images, conf_threshold):
    """处理图片列表，返回结果数据和生成的文件"""
    results_data = []
    result_images = {}  # 用于存储结果图片的字典 {文件名: 图片数据}
    progress_bar = st.progress(0, text="开始处理...")
    status_text = st.empty()

    for idx, (name, img_bytes) in enumerate(image_files.items()):
        status_text.text(f"正在处理 [{idx+1}/{len(image_files)}]: {name}")
        progress_bar.progress((idx + 1) / len(image_files))

        # 读取图片
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # 推理
        results = model(img, conf=conf_threshold)
        boxes = results[0].boxes

        detected = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x_center = float(box.xywh[0][0])
                detected.append((x_center, cls, conf))

        if not detected:
            full_number = 'N/A'
            avg_conf = 0.0
        else:
            detected.sort(key=lambda x: x[0])
            digits = [str(d[1]) for d in detected]
            confidences = [d[2] for d in detected]
            full_number = ''.join(digits)
            avg_conf = sum(confidences) / len(confidences)

        # 提取时间
        frame_num = extract_frame_number(name)
        time_sec = frame_num if frame_num is not None else idx

        results_data.append([time_sec, full_number, f"{avg_conf:.3f}"])

        # 保存结果图片
        if save_images and detected:
            annotated_img = results[0].plot()
            is_success, buffer = cv2.imencode(".jpg", annotated_img)
            if is_success:
                result_images[f"result_{time_sec:04d}_{name}"] = buffer.tobytes()

    status_text.text("✅ 处理完成！")
    progress_bar.empty()
    return results_data, result_images

# ========== 5. 主逻辑 ==========
if uploaded_zip is not None:
    # 解压 ZIP 文件到内存
    with st.spinner("📦 正在解压 ZIP 文件..."):
        image_files = {}
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zip_ref:
            for file_info in zip_ref.infolist():
                # 跳过目录和非图片文件
                if file_info.is_dir():
                    continue
                # 检查扩展名
                ext = Path(file_info.filename).suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    try:
                        image_files[file_info.filename] = zip_ref.read(file_info.filename)
                    except Exception as e:
                        st.warning(f"无法读取文件: {file_info.filename}, 错误: {e}")

    if not image_files:
        st.error("❌ ZIP 包中未找到任何支持的图片文件。")
        st.stop()

    st.info(f"📁 共找到 {len(image_files)} 张图片")

    # 处理图片
    results_data, result_images = process_images(
        image_files, model, save_images, conf_threshold
    )

    if not results_data:
        st.error("❌ 未能识别出任何有效数据。")
        st.stop()

    # ========== 6. 显示结果预览 ==========
    st.subheader("📊 识别结果预览")
    df = pd.DataFrame(results_data, columns=['时间(s)', '电动势', '置信度'])
    st.dataframe(df.head(20), use_container_width=True)

    # ========== 7. 生成并显示曲线图 ==========
    if generate_plot and len(results_data) > 1:
        st.subheader("📈 电动势-时间平滑曲线")
        try:
            # 过滤有效数据
            valid_data = [row for row in results_data if row[1] != 'N/A']
            if len(valid_data) >= 4:
                times = [float(row[0]) for row in valid_data]
                emfs = [float(row[1]) for row in valid_data]
                confs = [float(row[2]) for row in valid_data]

                # 转为numpy并排序
                times = np.array(times)
                emfs = np.array(emfs)
                confs = np.array(confs)
                sort_idx = np.argsort(times)
                times_sorted = times[sort_idx]
                emfs_sorted = emfs[sort_idx]
                confs_sorted = confs[sort_idx]

                # 过滤异常值
                filter_mask = (emfs_sorted >= 100) & (emfs_sorted <= 1000)
                times_plot = times_sorted[filter_mask]
                emfs_plot = emfs_sorted[filter_mask]
                confs_plot = confs_sorted[filter_mask]

                if len(times_plot) >= 4:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    # 图1：原始数据点
                    scatter = ax1.scatter(times_plot, emfs_plot, c=confs_plot, cmap='viridis', s=20, alpha=0.6)
                    ax1.plot(times_plot, emfs_plot, 'b--', alpha=0.3)
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('EMF (mV)')
                    ax1.set_title('EMF vs Time - Raw Data')
                    ax1.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax1, label='Confidence')

                    # 图2：平滑曲线
                    x_smooth = np.linspace(times_plot.min(), times_plot.max(), 300)
                    spl = make_interp_spline(times_plot, emfs_plot, k=min(3, len(times_plot)-1))
                    y_smooth = spl(x_smooth)
                    ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smooth Curve')
                    ax2.scatter(times_plot, emfs_plot, color='blue', s=20, label='Raw Data')
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('EMF (mV)')
                    ax2.set_title('EMF vs Time - Smooth Curve')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("过滤后有效数据点不足，无法生成平滑曲线。")
            else:
                st.warning("有效数据点不足（至少需要4个），无法生成平滑曲线。")
        except Exception as e:
            st.warning(f"生成曲线图时出错: {e}")

    # ========== 8. 提供下载 ==========
    st.subheader("📥 下载结果")

    # 8.1 下载 CSV
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(['时间(s)', '电动势', '置信度'])
    writer.writerows(results_data)
    st.download_button(
        label="📊 下载 CSV 结果",
        data=csv_buffer.getvalue(),
        file_name="recognition_results.csv",
        mime="text/csv"
    )

    # 8.2 下载结果图片 ZIP
    if result_images:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_out:
            for fname, data in result_images.items():
                zip_out.writestr(fname, data)
        st.download_button(
            label="🖼️ 下载结果图片 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="result_images.zip",
            mime="application/zip"
        )

    st.success("🎉 所有任务完成！")