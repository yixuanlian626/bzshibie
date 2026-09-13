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
from pathlib import Path
import pandas as pd
import hashlib

# ========== 1. 页面配置 ==========
st.set_page_config(page_title="数码管批量识别", layout="wide")
st.title("📟 数码管数字批量识别工具")
st.markdown("上传包含数码管图片的 **ZIP 压缩包** 或 **视频文件**，系统将自动识别所有图片中的数字组合并生成 CSV 结果。")

# ========== 初始化 session_state ==========
if 'raw_results_data' not in st.session_state:
    st.session_state.raw_results_data = None
if 'frame_images' not in st.session_state:
    st.session_state.frame_images = {}
if 'processed_file_hash' not in st.session_state:
    st.session_state.processed_file_hash = None

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
    st.stop()

# ========== 3. 侧边栏：参数设置 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    temperature = st.number_input(
        "🌡️ 温度 (摄氏度)",
        value=25.00,
        step=0.01,
        format="%.2f",
        help="输入当前实验温度（四位有效数字，两位小数），将用于CSV文件命名"
    )
    
    input_type = st.radio(
        "选择输入类型",
        ["📁 图片压缩包 (ZIP)", "🎬 视频文件"],
        index=0
    )
    
    fps_choice = None
    if input_type == "🎬 视频文件":
        st.subheader("🎞️ 抽帧设置")
        fps_choice = st.selectbox(
            "抽帧频率 (每秒帧数)",
            options=[0.5, 1, 2, 5, 10, 15, 30],
            index=1,
            format_func=lambda x: f"{x} 帧/秒" if x != 0.5 else "每2秒1帧"
        )
    
    save_frames = st.checkbox("保存抽帧原图（仅视频模式）", value=True) if input_type == "🎬 视频文件" else False
    generate_plot = st.checkbox("生成电动势-时间平滑曲线图", value=True)
    
    st.subheader("🎯 置信度设置")
    conf_threshold = st.slider(
        "模型推理置信度阈值",
        0.0, 1.0, 0.25, 0.05,
        help="传给YOLO模型的置信度，控制模型输出哪些检测框"
    )
    filter_conf_threshold = st.number_input(
        "结果筛选置信度阈值",
        value=0.50,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        help="在模型输出结果上再筛选：只保留置信度 >= 该值的检测结果，低于此值的结果将被剔除"
    )

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

def process_images(image_files, model, conf_threshold, batch_size=8):
    """处理图片列表（批量推理版）
    返回 raw_results_data: [(time_sec, raw_detections), ...]
    raw_detections: [(x_center, cls, conf), ...]
    """
    raw_results_data = []
    
    items = list(image_files.items())
    total = len(items)
    
    progress_bar = st.progress(0, text="开始处理...")
    status_text = st.empty()
    
    for start in range(0, total, batch_size):
        batch_items = items[start:start + batch_size]
        batch_names = [name for name, _ in batch_items]
        batch_imgs = []
        
        for name, img_data in batch_items:
            if isinstance(img_data, bytes):
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = img_data
            batch_imgs.append(img)
        
        valid_indices = [i for i, img in enumerate(batch_imgs) if img is not None]
        valid_imgs = [batch_imgs[i] for i in valid_indices]
        valid_names = [batch_names[i] for i in valid_indices]
        
        if not valid_imgs:
            continue
        
        results = model(valid_imgs, conf=conf_threshold)
        
        for idx, (name, result) in enumerate(zip(valid_names, results)):
            boxes = result.boxes
            detected = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x_center = float(box.xywh[0][0])
                    detected.append((x_center, cls, conf))
            
            frame_num = extract_frame_number(name)
            time_sec = frame_num if frame_num is not None else start + idx
            # 保存原始检测列表（未筛选）
            raw_results_data.append((time_sec, detected))
        
        progress_bar.progress(min((start + batch_size) / total, 1.0))
        status_text.text(f"已处理 {min(start + batch_size, total)}/{total} 张")
    
    status_text.text("✅ 处理完成！")
    progress_bar.empty()
    return raw_results_data

def process_video(video_bytes, model, fps, save_frames, conf_threshold):
    """处理视频：流式抽帧 + 即时识别（内存友好）
    返回 raw_results_data 和 frame_images
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0
    
    precise_frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        precise_frame_count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = precise_frame_count
    
    frame_interval = 1 if fps >= video_fps else int(video_fps / fps)
    
    raw_results_data = []
    frame_images = {}
    
    frame_count = 0
    extracted_count = 0
    
    progress_bar = st.progress(0, text="正在抽帧并识别...")
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            status_text.text(f"处理中: {frame_count}/{total_frames} (间隔 {frame_interval} 帧)")
            progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)
            
            time_sec = int(frame_count / video_fps)
            
            results = model(frame, conf=conf_threshold)
            boxes = results[0].boxes
            
            detected = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x_center = float(box.xywh[0][0])
                    detected.append((x_center, cls, conf))
            
            raw_results_data.append((time_sec, detected))
            extracted_count += 1
            
            if save_frames:
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    frame_images[f"original_{time_sec:04d}.jpg"] = buffer.tobytes()
                    if len(frame_images) > 200:
                        oldest_key = list(frame_images.keys())[0]
                        del frame_images[oldest_key]
        
        frame_count += 1
    
    cap.release()
    os.unlink(tmp_path)
    
    st.info(f"📁 从视频中抽取并识别了 {extracted_count} 帧图片")
    
    if len(frame_images) >= 200:
        st.warning("⚠️ 为节省内存，抽帧原图仅保留最后200张。如需全部图片，请使用图片压缩包模式。")
    
    return raw_results_data, frame_images

def apply_filter(raw_results_data, filter_conf_threshold):
    """按筛选置信度阈值重新计算识别结果
    返回 results_data: [[time_sec, full_number, avg_conf], ...]
    """
    results_data = []
    for time_sec, raw_detections in raw_results_data:
        filtered = [d for d in raw_detections if d[2] >= filter_conf_threshold]
        
        if not filtered:
            full_number = 'N/A'
            avg_conf = 0.0
        else:
            filtered.sort(key=lambda x: x[0])
            digits = [str(d[1]) for d in filtered]
            confidences = [d[2] for d in filtered]
            full_number = ''.join(digits)
            avg_conf = sum(confidences) / len(confidences)
        
        results_data.append([time_sec, full_number, f"{avg_conf:.3f}"])
    return results_data

# ========== 5. 主逻辑：根据输入类型分发 ==========
# ===== 5.1 图片压缩包模式 =====
if input_type == "📁 图片压缩包 (ZIP)":
    uploaded_file = st.file_uploader(
        "上传图片压缩包 (ZIP)",
        type=['zip'],
        help="请将图片打包成 ZIP 格式上传"
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        if file_hash != st.session_state.processed_file_hash:
            with st.spinner("📦 正在解压 ZIP 文件..."):
                image_files = {}
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zip_ref:
                    for file_info in zip_ref.infolist():
                        if file_info.is_dir():
                            continue
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
            
            raw_results_data = process_images(image_files, model, conf_threshold)
            
            st.session_state.raw_results_data = raw_results_data
            st.session_state.frame_images = {}
            st.session_state.processed_file_hash = file_hash
        else:
            st.info("📁 使用已缓存的识别结果（修改筛选阈值或点击下载不会重新识别）")

# ===== 5.2 视频模式 =====
else:
    uploaded_video = st.file_uploader(
        "上传视频文件",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
        help="支持的格式: MP4, AVI, MOV, MKV, FLV, WMV"
    )
    
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        file_hash = hashlib.md5(video_bytes).hexdigest()
        
        if file_hash != st.session_state.processed_file_hash:
            try:
                raw_results_data, frame_images = process_video(
                    video_bytes, model, fps_choice, save_frames, conf_threshold
                )
                
                st.session_state.raw_results_data = raw_results_data
                st.session_state.frame_images = frame_images
                st.session_state.processed_file_hash = file_hash
            except Exception as e:
                st.error(f"❌ 处理视频时出错: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
        else:
            st.info("🎬 使用已缓存的识别结果（修改筛选阈值或点击下载不会重新识别）")

# ========== 6. 显示与下载结果 ==========
raw_results_data = st.session_state.raw_results_data
frame_images = st.session_state.frame_images

if raw_results_data:
    # 按筛选置信度阈值重新计算
    results_data = apply_filter(raw_results_data, filter_conf_threshold)
    
    if not results_data:
        st.error("❌ 未能识别出任何有效数据。")
        st.stop()
    
    # ===== 6.1 显示结果预览 =====
    st.subheader("📊 识别结果预览")
    st.caption(f"当前筛选置信度阈值: {filter_conf_threshold:.2f}（低于此值的检测结果已被剔除）")
    df = pd.DataFrame(results_data, columns=['Time (s)', 'EMF (mV)', 'Confidence'])
    st.dataframe(df.head(20), use_container_width=True)
    
    valid_count = len([r for r in results_data if r[1] != 'N/A'])
    st.caption(f"有效识别: {valid_count} / {len(results_data)} 张")
    
    # ===== 6.2 显示曲线图 =====
    if generate_plot and len(results_data) > 1:
        st.subheader("📈 电动势-时间平滑曲线")
        try:
            valid_data = [row for row in results_data if row[1] != 'N/A']
            if len(valid_data) >= 4:
                times = [float(row[0]) for row in valid_data]
                emfs = [float(row[1]) for row in valid_data]
                confs = [float(row[2]) for row in valid_data]
                
                times = np.array(times)
                emfs = np.array(emfs)
                confs = np.array(confs)
                sort_idx = np.argsort(times)
                times_sorted = times[sort_idx]
                emfs_sorted = emfs[sort_idx]
                confs_sorted = confs[sort_idx]
                
                filter_mask = (emfs_sorted >= 100) & (emfs_sorted <= 1000)
                times_plot = times_sorted[filter_mask]
                emfs_plot = emfs_sorted[filter_mask]
                confs_plot = confs_sorted[filter_mask]
                
                if len(times_plot) >= 4:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    scatter = ax1.scatter(times_plot, emfs_plot, c=confs_plot, cmap='viridis', s=20, alpha=0.6)
                    ax1.plot(times_plot, emfs_plot, 'b--', alpha=0.3)
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('EMF (mV)')
                    ax1.set_title('EMF vs Time - Raw Data')
                    ax1.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax1, label='Confidence')
                    
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
    
    # ===== 6.3 下载结果 =====
    st.subheader("📥 下载结果")
    
    temp_str = f"{temperature:.2f}"
    conf_str = f"{filter_conf_threshold:.2f}"
    csv_filename = f"{temp_str}_conf{conf_str}.csv"
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(['Time (s)', 'EMF (mV)', 'Confidence'])
        writer.writerows(results_data)
        st.download_button(
            label="📊 下载 CSV 结果",
            data=csv_buffer.getvalue(),
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if frame_images:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_out:
                for fname, data in frame_images.items():
                    zip_out.writestr(fname, data)
            zip_filename = f"frames_{temp_str}.zip"
            st.download_button(
                label="🖼️ 下载抽帧原图 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.button("🖼️ 下载抽帧原图 (无)", disabled=True, use_container_width=True)
    
    st.success("🎉 所有任务完成！")
