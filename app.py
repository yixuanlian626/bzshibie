import streamlit as st
from ultralytics import YOLO
import cv2
import os
import csv
import io
import gc
import zipfile
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import re
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pathlib import Path
import pandas as pd
import hashlib

# ========== 1. 页面配置（界面完全保留原版） ==========
st.set_page_config(page_title="数码管批量识别", layout="wide")
st.title("📟 数码管数字批量识别工具")
st.markdown("上传包含数码管图片的 **ZIP 压缩包** 或 **视频文件**，系统将自动识别所有图片中的数字组合并生成 CSV 结果。")

# ========== session_state缓存（新增，文件缓存避免重复推理） ==========
if 'raw_results_data' not in st.session_state:
    st.session_state.raw_results_data = None
if 'result_images' not in st.session_state:
    st.session_state.result_images = {}
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

# ========== 3. 侧边栏：参数设置【完全保留原版界面，不修改】 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")

    # 温度输入 - 四位有效数字，两位小数
    temperature = st.number_input(
        "🌡️ 温度 (摄氏度)",
        value=25.00,
        step=0.01,
        format="%.2f",
        help="输入当前实验温度（四位有效数字，两位小数），将用于CSV文件命名"
    )

    # 输入源选择
    input_type = st.radio(
        "选择输入类型",
        ["📁 图片压缩包 (ZIP)", "🎬 视频文件"],
        index=0
    )

    # 视频抽帧参数（仅在视频模式下显示）
    fps_choice = None
    if input_type == "🎬 视频文件":
        st.subheader("🎞️ 抽帧设置")
        fps_choice = st.selectbox(
            "抽帧频率 (每秒帧数)",
            options=[0.5, 1, 2, 5, 10, 15, 30],
            index=1,  # 默认 1 fps
            format_func=lambda x: f"{x} 帧/秒" if x != 0.5 else "每2秒1帧"
        )

    # 通用参数【原版UI完全不动】
    save_images = st.checkbox("保存带检测框的结果图片", value=True)
    save_frames = st.checkbox("保存抽帧原图（仅视频模式）", value=True) if input_type == "🎬 视频文件" else False
    generate_plot = st.checkbox("生成电动势-时间平滑曲线图", value=True)
    conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05)

    # 新增性能参数，放在侧边栏最下面，不改动原有UI视觉
    st.divider()
    st.subheader("性能设置")
    batch_size = st.selectbox("批处理大小", options=[2, 4, 8, 16], index=1)
    use_half_res = st.checkbox("半分辨率解码（加速）", value=True)


# ========== 工具函数 ==========
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


def stream_hash(fileobj, chunk_size=1024 * 1024):
    """流计算md5，不全部读入内存"""
    hasher = hashlib.md5()
    fileobj.seek(0)
    while chunk := fileobj.read(chunk_size):
        hasher.update(chunk)
    fileobj.seek(0)
    return hasher.hexdigest()


def process_zip_stream_memory_safe(zip_temp_path, model, conf_threshold, save_images, batch_size=4, use_half_res=True, ui_interval=20):
    """
    内存安全ZIP处理：不预加载全部图片二进制，batch批量推理
    返回 results_data, result_images
    """
    raw_results_data = []
    result_images = {}
    image_suffix = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    with zipfile.ZipFile(zip_temp_path) as zip_ref:
        # 只收集文件信息，不读取图片字节
        image_infos = []
        for info in zip_ref.infolist():
            if not info.is_dir() and Path(info.filename).suffix.lower() in image_suffix:
                image_infos.append(info)
        total = len(image_infos)
        if total == 0:
            return [], {}

        progress_bar = st.progress(0, text="开始处理...")
        status_text = st.empty()

        batch_names = []
        batch_imgs = []
        processed_count = 0

        for idx, info in enumerate(image_infos):
            try:
                img_bytes = zip_ref.read(info.filename)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                del img_bytes, nparr
            except Exception:
                img = None

            if img is not None:
                if use_half_res:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                batch_names.append(info.filename)
                batch_imgs.append(img)

            # batch执行推理
            if len(batch_imgs) >= batch_size or idx == total - 1:
                if batch_imgs:
                    results = model(batch_imgs, conf=conf_threshold, verbose=False)
                    for name, res in zip(batch_names, results):
                        boxes = res.boxes
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

                        frame_num = extract_frame_number(name)
                        time_sec = frame_num if frame_num is not None else processed_count
                        raw_results_data.append([time_sec, full_number, f"{avg_conf:.3f}"])

                        # 保存带标注图片；限制内存最多200张，防止OOM
                        if save_images and detected:
                            annotated_img = res.plot()
                            ok, buf = cv2.imencode(".jpg", annotated_img)
                            if ok:
                                result_images[f"result_{time_sec:04d}_{name}"] = buf.tobytes()
                                if len(result_images) > 200:
                                    oldest = list(result_images.keys())[0]
                                    del result_images[oldest]
                        processed_count += 1

                    # 释放batch内存
                    for im in batch_imgs:
                        del im
                    del batch_imgs, results
                    batch_names = []
                    batch_imgs = []
                    gc.collect()

                # UI间隔刷新，减少重渲染
                if (idx + 1) % ui_interval == 0 or (idx + 1) == total:
                    progress_bar.progress((idx + 1) / total)
                    status_text.text(f"正在处理 [{idx+1}/{total}]")

        status_text.text("✅ 处理完成！")
        progress_bar.empty()
    return raw_results_data, result_images


def process_video(video_bytes, model, fps, save_images, save_frames, conf_threshold):
    """完全保留原版视频处理逻辑，仅小量内存优化"""
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

    results_data = []
    result_images = {}
    frame_images = {}

    frame_count = 0
    extracted_count = 0

    progress_bar = st.progress(0, text="正在抽帧并识别...")
    status_text = st.empty()

    save_original_frames = save_frames
    save_result_images = save_images

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            status_text.text(f"处理中: {frame_count}/{total_frames} (间隔 {frame_interval} 帧)")
            progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)

            time_sec = int(frame_count / video_fps)
            filename = f"frame_{time_sec:04d}.jpg"

            results = model(frame, conf=conf_threshold, verbose=False)
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

            results_data.append([time_sec, full_number, f"{avg_conf:.3f}"])
            extracted_count += 1

            if save_result_images and detected:
                annotated_img = results[0].plot()
                is_success, buffer = cv2.imencode(".jpg", annotated_img)
                if is_success:
                    result_images[f"result_{time_sec:04d}.jpg"] = buffer.tobytes()
                    if len(result_images) > 200:
                        oldest_key = list(result_images.keys())[0]
                        del result_images[oldest_key]

            if save_original_frames:
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
    if len(result_images) >= 200 or len(frame_images) >= 200:
        st.warning("⚠️ 为节省内存，结果图片仅保留最后200张。如需全部图片，请使用图片压缩包模式。")

    return results_data, result_images, frame_images


# ========== 5. 主逻辑：根据输入类型分发 ==========
if input_type == "📁 图片压缩包 (ZIP)":
    uploaded_file = st.file_uploader(
        "上传图片压缩包 (ZIP)",
        type=['zip'],
        help="请将图片打包成 ZIP 格式上传"
    )

    if uploaded_file is not None:
        # === 关键：立刻显示解压提示，复刻旧版本交互感受 ===
        with st.spinner("📦 正在解压 ZIP 文件..."):
            # 流计算hash
            file_hash = stream_hash(uploaded_file)
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"ZIP文件大小: {file_size_mb:.1f} MB")
            if file_size_mb > 300:
                st.warning("⚠️ 文件较大，处理会消耗较多时间，请耐心等待。")

        # 缓存命中，直接读取session_state
        if file_hash == st.session_state.processed_file_hash:
            st.info("📁 使用已缓存的识别结果（修改筛选阈值或点击下载不会重新识别）")
        else:
            # 分块写入临时磁盘文件，不会一次性读全部zip进内存
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                uploaded_file.seek(0)
                while chunk := uploaded_file.read(8 * 1024 * 1024):
                    tmp_zip.write(chunk)
                tmp_zip_path = tmp_zip.name

            st.info("📦 正在流式解压并识别...")
            try:
                results_data, result_images = process_zip_stream_memory_safe(
                    tmp_zip_path,
                    model,
                    conf_threshold=conf_threshold,
                    save_images=save_images,
                    batch_size=batch_size,
                    use_half_res=use_half_res
                )
            finally:
                try:
                    os.unlink(tmp_zip_path)
                except Exception:
                    pass

            if not results_data:
                st.error("❌ ZIP 包中未找到任何支持的图片文件。")
                st.stop()

            # 存入缓存
            st.session_state.raw_results_data = results_data
            st.session_state.result_images = result_images
            st.session_state.frame_images = {}
            st.session_state.processed_file_hash = file_hash

        # 从缓存取出数据
        results_data = st.session_state.raw_results_data
        result_images = st.session_state.result_images
        frame_images = st.session_state.frame_images

else:
    uploaded_video = st.file_uploader(
        "上传视频文件",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
        help="支持的格式: MP4, AVI, MOV, MKV, FLV, WMV"
    )

    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        file_hash = hashlib.md5(video_bytes).hexdigest()
        if file_hash == st.session_state.processed_file_hash:
            st.info("🎬 使用已缓存的识别结果")
            results_data = st.session_state.raw_results_data
            result_images = st.session_state.result_images
            frame_images = st.session_state.frame_images
        else:
            try:
                results_data, result_images, frame_images = process_video(
                    video_bytes,
                    model,
                    fps_choice,
                    save_images,
                    save_frames,
                    conf_threshold
                )
                st.session_state.raw_results_data = results_data
                st.session_state.result_images = result_images
                st.session_state.frame_images = frame_images
                st.session_state.processed_file_hash = file_hash
            except Exception as e:
                st.error(f"❌ 处理视频时出错: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
            finally:
                del video_bytes
                gc.collect()


# ========== 6. 显示与下载结果【UI完全和原版保持原样，不修改】 ==========
results_data = st.session_state.get("raw_results_data")
result_images = st.session_state.get("result_images", {})
frame_images = st.session_state.get("frame_images", {})

if results_data:
    if not results_data:
        st.error("❌ 未能识别出任何有效数据。")
        st.stop()

    # ===== 6.1 显示结果预览 =====
    st.subheader("📊 识别结果预览")
    df = pd.DataFrame(results_data, columns=['Time (s)', 'EMF (mV)', 'Confidence'])
    st.dataframe(df.head(20), use_container_width=True)

    # 统计信息
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
                    spl = make_interp_spline(times_plot, emfs_plot, k=min(3, len(times_plot) - 1))
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
                    plt.close(fig)
                else:
                    st.warning("过滤后有效数据点不足，无法生成平滑曲线。")
            else:
                st.warning("有效数据点不足（至少需要4个），无法生成平滑曲线。")
        except Exception as e:
            st.warning(f"生成曲线图时出错: {e}")

    # ===== 6.3 下载结果【原版三列下载按钮完全保留】 =====
    st.subheader("📥 下载结果")
    temp_str = f"{temperature:.2f}"
    csv_filename = f"{temp_str}.csv"

    col1, col2, col3 = st.columns(3)

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
        if result_images:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_out:
                for fname, data in result_images.items():
                    zip_out.writestr(fname, data)
            zip_filename = f"{temp_str}.zip"
            st.download_button(
                label="🖼️ 下载结果图片 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.button("🖼️ 下载结果图片 (无)", disabled=True, use_container_width=True)

    with col3:
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
