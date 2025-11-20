import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
# 核心改动：引入 face_recognition，移除 mediapipe
import face_recognition 

# --- 页面配置 ---
st.set_page_config(
    page_title="MyFaceOnly - 隐私打码助手",
    page_icon="🫣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义 CSS (美化 UI) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .face-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background: white;
        text-align: center;
        margin-bottom: 10px;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 核心功能函数 ---

# 移除 @st.cache_resource，因为 face_recognition 库无需初始化检测器对象
# @st.cache_resource
# def get_face_detector():
#     mp_face_detection = mp.solutions.face_detection
#     return mp_face_detection.FaceDetec    tion(model_selection=1, min_detection_confidence=0.5)

def process_image(image, blur_strength, style, face_visibility_states, detections):
    """核心图像处理逻辑：根据状态对人脸进行打码，使用圆形柔和过渡。"""
    img_np = np.array(image)
    
    h, w, _ = img_np.shape
    output_img = img_np.copy()
    
    # detections 现在是一个列表，元素格式为 (top, right, bottom, left)
    for i, bbox in enumerate(detections):
        # 如果该人脸被标记为“显示”(True)，则跳过处理
        if face_visibility_states.get(i, False):
            continue

        # 将 face_recognition 的坐标 (top, right, bottom, left)
        top, right, bottom, left = bbox
        
        # 增加一个 padding 让模糊区域比检测框略大，提高美观度
        padding = 10
        
        x = max(0, left - padding)
        y = max(0, top - padding)
        x_end = min(w, right + padding)
        y_end = min(h, bottom + padding)
        
        # 重新计算带 padding 的 ROI 尺寸
        w_box = x_end - x
        h_box = y_end - y
        
        if w_box <= 0 or h_box <= 0: continue

        # 提取人脸区域 (ROI)
        roi = output_img[y:y+h_box, x:x+w_box]
        
        if roi.size == 0: continue

        # --- 核心改动 1: 应用打码效果到 ROI ---
        if style == "毛玻璃 (Gaussian Blur)":
            # --- 修复 ZeroDivisionError 的逻辑 ---
            # 使用反向映射来计算，确保强度100对应最大模糊
            # 强度 10 对应分母 91 (小 ksize)，强度 100 对应分母 1 (大 ksize)
            # 确保分母至少为 1
            denominator = max(1, 101 - blur_strength)
            
            # ksize_val: 决定模糊核大小。乘数 10 调整模糊与人脸尺寸的比例。
            ksize_val = int(w_box / denominator * 10) 
            
            # 限制 ksize 的最大值（防止性能问题），并确保最小值
            ksize_val = min(49, max(3, ksize_val)) 
            
            # 确保 ksize 是奇数
            ksize = ksize_val | 1 
            
            processed_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            
        elif style == "马赛克 (Mosaic)":
            # 缩小再放大实现马赛克
            # 强度 10 对应 pixel_size 小 (清晰)，强度 100 对应 pixel_size 大 (模糊)
            # max(1, ...) 确保 pixel_size 至少为 1
            pixel_size = max(1, int(w_box // (100 / blur_strength * 3)))

            roi_small = cv2.resize(roi, 
                                   (max(1, w_box // pixel_size), max(1, h_box // pixel_size)), 
                                   interpolation=cv2.INTER_LINEAR)
            processed_roi = cv2.resize(roi_small, 
                                       (w_box, h_box), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # --- 核心改动 2: 创建和应用圆形柔和遮罩 ---
        
        # 1. 创建一个单通道的零矩阵作为遮罩
        mask = np.zeros((h_box, w_box), dtype=np.float32)
        
        # 2. 计算人脸区域的中心点和半径
        center_x, center_y = w_box // 2, h_box // 2
        # 半径取较小边的一半的90%
        radius = min(center_x, center_y) * 0.9 
        
        # 3. 使用 cv2.circle 绘制实心白圆 (值设为 255)
        cv2.circle(mask, (center_x, center_y), int(radius), (255), -1)

        # 4. 对圆形掩码进行高斯模糊，实现柔和过渡 (关键步骤)
        # sigma 与半径关联，确保过渡自然。至少为 3。
        sigma = max(3, int(radius * 0.15)) 
        
        # 核大小取最大的边长，保证边缘过渡足够平滑。确保是奇数。
        blur_ksize = (w_box | 1, h_box | 1) 
        mask_blurred = cv2.GaussianBlur(mask, blur_ksize, sigmaX=sigma)
        
        # 5. 归一化到 0-1 范围，并确保是 3 通道 (与图像 ROIs 尺寸匹配)
        mask_float = mask_blurred / 255.0
        mask_3channel = np.stack([mask_float] * 3, axis=-1)
        
        # 6. 合并：使用 alpha 混合公式实现柔和过渡
        # output_img_roi = processed_roi * mask_3channel + roi * (1 - mask_3channel)
        output_img_roi = cv2.addWeighted(processed_roi.astype(np.float32), 
                                         1.0, 
                                         roi.astype(np.float32), 
                                         0.0, 
                                         0.0)
        # 用原图与圆形打码区域进行混合
        output_img_roi = output_img_roi * mask_3channel + roi.astype(np.float32) * (1 - mask_3channel)
        
        output_img[y:y+h_box, x:x+w_box] = output_img_roi.astype(np.uint8)

    return output_img

# --- 主界面逻辑 ---

def main():
    # 侧边栏：设置
    st.sidebar.title("🛠️ 设置工具箱")
    
    # 上传组件
    uploaded_file = st.sidebar.file_uploader("1. 上传照片", type=['jpg', 'jpeg', 'png'])
    
    # 打码风格设置
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 打码风格")
    blur_style = st.sidebar.radio("选择样式", ["毛玻璃 (Gaussian Blur)", "马赛克 (Mosaic)"])
    blur_strength = st.sidebar.slider("模糊强度", min_value=10, max_value=100, value=60, step=5)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 说明：勾选下方**'保留'**的人脸将保持清晰，未勾选的将被自动打码。")

    # 标题区
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🫣 MyFaceOnly")
        st.markdown("#### 只露我脸 - 智能隐私保护工具")
    
    if uploaded_file is not None:
        # 读取图片
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        
        # 初始化 Session State (防止刷新重置)
        if 'detections' not in st.session_state or st.session_state.get('last_uploaded') != uploaded_file.name:
            with st.spinner('正在使用高精度模型识别图中人像...'):
                # --- 核心改动：使用 face_recognition 进行人脸检测 ---
                # face_locations 返回一个列表，每个元素是人脸的 (top, right, bottom, left) 坐标
                face_locations = face_recognition.face_locations(img_np, model="hog") 
                
                st.session_state['detections'] = face_locations
                st.session_state['last_uploaded'] = uploaded_file.name
                # 默认所有人都打码 (False)，用户手动选择自己 (True)
                st.session_state['face_states'] = {i: False for i in range(len(st.session_state['detections']))}

        detections = st.session_state['detections']
        
        if not detections:
            st.warning("未检测到任何人脸，请尝试更换清晰的照片。")
            st.image(image, use_container_width=True)
        else:
            # --- 人脸选择区 (交互核心) ---
            st.markdown(f"##### 📸 检测到 {len(detections)} 张人脸")
            st.caption("请勾选 **你自己** (或你想保留清晰的人脸)")

            # 使用 expander 收纳人脸选择器，避免占用过多垂直空间
            with st.expander("👤 点击此处展开/折叠 人脸选择面板", expanded=True):
                # 动态创建列来展示人脸缩略图
                cols_per_row = 5
                cols = st.columns(cols_per_row)
                
                h_img, w_img, _ = img_np.shape
                
                for i, bbox in enumerate(detections):
                    # 获取 face_recognition 的坐标 (top, right, bottom, left)
                    top, right, bottom, left = bbox
                    
                    # 稍微扩大一点截图范围，更好辨认
                    pad = 20
                    y1, y2 = max(0, top - pad), min(h_img, bottom + pad)
                    x1, x2 = max(0, left - pad), min(w_img, right + pad)
                    face_thumb = img_np[y1:y2, x1:x2]
                    
                    # 在对应的列中显示
                    col_idx = i % cols_per_row
                    with cols[col_idx]:
                        st.image(face_thumb, use_container_width=True)
                        # Checkbox 绑定状态
                        is_visible = st.checkbox(f"保留 #{i+1}", value=st.session_state['face_states'][i], key=f"face_{i}")
                        st.session_state['face_states'][i] = is_visible

            # --- 实时预览与处理 ---
            st.markdown("### ✨ 效果预览")
            
            # 处理图片
            processed_img_np = process_image(
                image, 
                blur_strength, 
                blur_style, 
                st.session_state['face_states'], 
                detections
            )
            
            processed_img_pil = Image.fromarray(processed_img_np)
            st.image(processed_img_pil, use_container_width=True, caption="处理后的图片")

            # --- 下载区域 ---
            st.markdown("### 💾 保存结果")
            buf = io.BytesIO()
            processed_img_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            col_dl1, col_dl2 = st.columns([1, 3])
            with col_dl1:
                st.download_button(
                    label="📥 下载处理后的图片",
                    data=byte_im,
                    file_name="myfaceonly_result.png",
                    mime="image/png",
                )

    else:
        # 欢迎页引导
        st.info("👈 请在左侧侧边栏上传一张照片开始使用。")
        st.markdown("""
        **MyFaceOnly 特点：**
        * 🛡️ **隐私优先**：所有处理均在本地运行，照片不会被保存。
        * 🎯 **精准控制**：识别每一张脸，由你决定谁露脸。
        * 🎨 **自然美观**：提供柔和的毛玻璃特效，保留照片氛围。
        """)

if __name__ == "__main__":
    main()