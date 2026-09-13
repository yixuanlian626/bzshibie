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
    
    st.subheader("⚡ 性能设置")
    batch_size = st.selectbox(
        "批处理大小",
        options=[8, 16, 32, 64],
        index=1,
        help="越大越快，但内存占用越高。1920×1080 JPEG 建议 16~32"
    )
    use_half_res = st.checkbox(
        "半分辨率解码（加速）",
        value=True,
        help="将 1920×1080 图片解码为 960×540，解码和推理更快；如识别精度下降可关闭"
    )
    imgsz = st.selectbox(
        "推理尺寸 (imgsz)",
        options=[512, 640, 736, 960],
        index=1,
        help="YOLO 推理时的输入尺寸，640 是常用平衡点"
    )
    
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
        help="对每张图片：计算所有检测数字的平均置信度，若平均置信度 < 该值，则整行删除；否则保留全部数字"
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

def process_zip_streaming(file_bytes, model, conf_threshold,
                          batch_size=16, use_half_res=True, imgsz=640):
    """流式从 ZIP 读取图片并批量推理，内存只驻留一个 batch。
    返回 raw_results_data: [(time_sec, raw_detections), ...]
    raw_detections: [(x_center, cls, conf), ...]
    """
    raw_results_data = []
    
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zip_ref:
        image_infos = [
            info for info in zip_ref.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower()
            in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        ]
        total = len(image_infos)
        
        if total == 0:
            return raw_results_data
        
        progress_bar = st.progress(0, text="开始处理...")
        status_text = st.empty()
        
        decode_flag = cv2.IMREAD_REDUCED_COLOR_2 if use_half_res else cv2.IMREAD_COLOR
        
        batch_names = []
        batch_imgs = []
        processed_count = 0
        global_idx = 0
        
        for info in image_infos:
            try:
                img_bytes = zip_ref.read(info.filename)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, decode_flag)
            except Exception:
                img = None
            
            if img is not None:
                batch_names.append(info.filename)
                batch_imgs.append(img)
            
            global_idx += 1
            
            if len(batch_imgs) >= batch_size or global_idx == total:
                if batch_imgs:
                    results = model(
                        batch_imgs,
                        conf=conf_threshold,
                        imgsz=imgsz,
                        verbose=False
                    )
                    
                    for name, result in zip(batch_names, results):
                        boxes = result.boxes
                        detected = []
                        if boxes is not None and len(boxes) > 0:
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                x_center = float(box.xywh[0][0])
                                detected.append((x_center, cls, conf))
                        
                        frame_num = extract_frame_number(name)
                        time_sec = frame_num if frame_num is not None else processed_count
                        raw_results_data.append((time_sec, detected))
                        processed_count += 1
                    
                    del batch_imgs, results
                    batch_names = []
                    batch_imgs = []
                
                progress_bar.progress(min(global_idx / total, 1.0))
                status_text.text(f"已处理 {global_idx}/{total} 张")
        
        status_text.text("✅ 处理完成！")
        progress_bar.empty()
    
    return raw_results_data

def process_video(video_bytes, model, fps, save_frames, conf_threshold,
                  batch_size=16, imgsz=640):
    """处理视频：流式抽帧 + 批量识别"""
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
    
    batch_frames = []
    batch_times = []
    
    def flush_batch():
        nonlocal extracted_count
        if not batch_frames:
            return
        results = model(batch_frames, conf=conf_threshold, imgsz=imgsz, verbose=False)
        for t_sec, result in zip(batch_times, results):
            boxes = result.boxes
            detected = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x_center = float(box.xywh[0][0])
                    detected.append((x_center, cls, conf))
            raw_results_data.append((t_sec, detected))
            extracted_count += 1
        batch_frames.clear()
        batch_times.clear()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            time_sec = int(frame_count / video_fps)
            batch_frames.append(frame)
            batch_times.append(time_sec)
            
            if save_frames:
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    frame_images[f"original_{time_sec:04d}.jpg"] = buffer.tobytes()
                    if len(frame_images) > 200:
                        oldest_key = list(frame_images.keys())[0]
                        del frame_images[oldest_key]
            
            if len(batch_frames) >= batch_size:
                flush_batch()
                progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)
                status_text.text(f"处理中: {frame_count}/{total_frames}")
        
        frame_count += 1
    
    flush_batch()
    
    cap.release()
    os.unlink(tmp_path)
    
    st.info(f"📁 从视频中抽取并识别了 {extracted_count} 帧图片")
    
    if len(frame_images) >= 200:
        st.warning("⚠️ 为节省内存，抽帧原图仅保留最后200张。")
    
    return raw_results_data, frame_images

def apply_filter(raw_results_data, filter_conf_threshold):
    """按筛选置信度阈值过滤：
    对每张图片，先计算所有检测数字的平均置信度，
    如果平均置信度 < filter_conf_threshold，则整行删除；
    否则保留该图片的完整识别结果（不剔除单个数字）。
    """
    results_data = []
    for time_sec, raw_detections in raw_results_data:
        if not raw_detections:
            # 该图片没有检测到任何数字，直接跳过（不写入CSV）
            continue
        
        # 计算这张图所有检测数字的平均置信度
        confidences = [d[2] for d in raw_detections]
        avg_conf = sum(confidences) / len(confidences)
        
        # 平均置信度低于阈值 → 整行删除
        if avg_conf < filter_conf_threshold:
            continue
        
        # 保留该图片的完整识别结果（所有数字都保留，不剔除单个数字）
        raw_detections_sorted = sorted(raw_detections, key=lambda x: x[0])
        digits = [str(d[1]) for d in raw_detections_sorted]
        full_number = ''.join(digits)
        
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
            st.info("📦 正在流式解压并识别...")
            raw_results_data = process_zip_streaming(
                file_bytes, model, conf_threshold,
                batch_size=batch_size,
                use_half_res=use_half_res,
                imgsz=imgsz
            )
            
            if not raw_results_data:
                st.error("❌ ZIP 包中未找到任何可处理的图片。")
                st.stop()
            
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
                    video_bytes, model, fps_choice, save_frames, conf_threshold,
                    batch_size=batch_size, imgsz=imgsz
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
    results_data = apply_filter(raw_results_data, filter_conf_threshold)
    
    if not results_data:
        st.warning("⚠️ 所有数据均被筛选掉（平均置信度均低于阈值），请降低筛选置信度阈值。")
        st.stop()
    
    # ===== 6.1 显示结果预览 =====
    st.subheader("📊 识别结果预览")
    st.caption(f"当前筛选置信度阈值: {filter_conf_threshold:.2f}（整张图平均置信度低于此值的行已被删除）")
    df = pd.DataFrame(results_data, columns=['Time (s)', 'EMF (mV)', 'Confidence'])
    st.dataframe(df.head(20), width='stretch')
    
    total_count = len(raw_results_data)
    kept_count = len(results_data)
    st.caption(f"保留: {kept_count} / {total_count} 张（删除 {total_count - kept_count} 张）")
    
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
    csv_filename = f"{temp_str}.csv"
    
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
            width='stretch'
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
                width='stretch'
            )
        else:
            st.button("🖼️ 下载抽帧原图 (无)", disabled=True, width='stretch')
    
    st.success("🎉 所有任务完成！")
