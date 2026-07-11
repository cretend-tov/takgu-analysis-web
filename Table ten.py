#!/usr/bin/env python3
"""
탁구 3D 분석 웹앱 (Streamlit 버전) v1.1
- 웹 브라우저에서 실행 (PC, 태블릿, 스마트폰 모두 지원)
- 영상 업로드 → 구간 선택 → 공 궤적 분석
- Top-down 및 3D 시각화 제공

실행 방법:
    pip install streamlit opencv-python pillow numpy matplotlib
    streamlit run 탁구_3D_분석_웹앱.py

배포 방법:
    - Streamlit Community Cloud (무료)
    - Hugging Face Spaces
    - 로컬 실행 후 핸드폰에서 같은 Wi-Fi로 접속 가능
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tempfile
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 페이지 설정
st.set_page_config(
    page_title="탁구 3D 분석 웹앱",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded"
)

TABLE_LENGTH_MM = 2740
TABLE_WIDTH_MM = 1525

@dataclass
class BallPosition:
    frame_idx: int
    pixel_x: int
    pixel_y: int
    world_x: Optional[float] = None
    world_y: Optional[float] = None

def detect_ball(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """흰색/주황색 공 검출"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 흰색 공
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 55, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # 주황색 공
    lower_orange = np.array([5, 90, 90])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    mask = cv2.bitwise_or(mask_white, mask_orange)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    best_center = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 12 or area > 9000:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.6 and area > best_score:
            best_score = area
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_center = (cx, cy)
    return best_center

def process_segment(video_path: str, start_sec: float, end_sec: float, 
                    fps: float, sample_step: int = 3) -> List[BallPosition]:
    """선택 구간에서 공 검출"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total = end_frame - start_frame
    
    positions = []
    frame_indices = list(range(start_frame, end_frame + 1, sample_step))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, f_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        center = detect_ball(frame)
        if center:
            positions.append(BallPosition(
                frame_idx=f_idx,
                pixel_x=center[0],
                pixel_y=center[1]
            ))
        
        progress = int((i + 1) / len(frame_indices) * 100)
        progress_bar.progress(progress)
        status_text.text(f"분석 중... {progress}% ({len(positions)}개 공 검출)")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    return positions

def create_topdown_plot(positions: List[BallPosition], 
                        homography: Optional[np.ndarray] = None) -> Figure:
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # 테이블 사각형
    table = plt.Rectangle((0, 0), TABLE_LENGTH_MM, TABLE_WIDTH_MM,
                          fill=False, edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(table)
    ax.axvline(TABLE_LENGTH_MM/2, color='#95a5a6', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(TABLE_WIDTH_MM/2, color='#95a5a6', linestyle='--', linewidth=1, alpha=0.7)
    
    if positions:
        if homography is not None:
            xs = [p.world_x for p in positions if p.world_x]
            ys = [p.world_y for p in positions if p.world_y]
            label = "공 궤적 (실제 mm)"
        else:
            xs = [p.pixel_x for p in positions]
            ys = [p.pixel_y for p in positions]
            label = "공 궤적 (픽셀)"
        
        if xs:
            scatter = ax.scatter(xs, ys, c=range(len(xs)), cmap='viridis', 
                                 s=30, alpha=0.85, zorder=3)
            ax.plot(xs, ys, color='#e74c3c', linewidth=1.8, alpha=0.7, zorder=2)
            fig.colorbar(scatter, ax=ax, label='시간 진행')
    
    ax.set_xlim(-150, TABLE_LENGTH_MM + 150)
    ax.set_ylim(-150, TABLE_WIDTH_MM + 150)
    ax.set_xlabel("길이 (mm)", fontsize=11)
    ax.set_ylabel("폭 (mm)", fontsize=11)
    ax.set_title("Top-down 궤적 분석", fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return fig

def create_3d_plot(positions: List[BallPosition],
                   homography: Optional[np.ndarray] = None) -> Figure:
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    if positions:
        if homography is not None:
            xs = np.array([p.world_x for p in positions if p.world_x])
            ys = np.array([p.world_y for p in positions if p.world_y])
            zs = np.zeros_like(xs)
        else:
            xs = np.array([p.pixel_x for p in positions])
            ys = np.array([p.pixel_y for p in positions])
            zs = np.zeros_like(xs)
        
        if len(xs) > 0:
            # 테이블 평면
            xx, yy = np.meshgrid(np.linspace(0, TABLE_LENGTH_MM, 8),
                                 np.linspace(0, TABLE_WIDTH_MM, 8))
            zz = np.zeros_like(xx)
            ax.plot_surface(xx, yy, zz, alpha=0.12, color='#3498db')
            
            # 테이블 테두리
            ax.plot([0, TABLE_LENGTH_MM, TABLE_LENGTH_MM, 0, 0],
                    [0, 0, TABLE_WIDTH_MM, TABLE_WIDTH_MM, 0],
                    [0]*5, color='#2c3e50', linewidth=2)
            
            # 공 궤적
            ax.plot(xs, ys, zs, color='#e74c3c', linewidth=2.5, label='공 궤적')
            ax.scatter(xs, ys, zs, c=range(len(xs)), cmap='plasma', s=35, alpha=0.9)
    
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D 공 궤적 시각화", fontsize=13, fontweight='bold')
    ax.view_init(elev=22, azim=-55)
    return fig

# ==================== Streamlit UI ====================
st.title("🏓 탁구 3D 분석 웹앱")
st.markdown("**웹 브라우저에서 실행되는 탁구 영상 분석 도구** — PC, 태블릿, 스마트폰 모두 지원")

with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader(
        "탁구 영상 업로드 (mp4, mov, avi)",
        type=["mp4", "mov", "avi", "mkv"],
        help="10~60초 정도의 랠리 영상이 가장 좋습니다."
    )
    
    st.markdown("---")
    st.info("📱 **스마트폰 사용 팁**\n"
            "크롬 또는 사파리 브라우저에서 접속하세요.\n"
            "영상은 가로 모드로 보는 것을 권장합니다.")

if not uploaded_file:
    st.warning("왼쪽 사이드바에서 탁구 영상 파일을 업로드해 주세요.")
    st.stop()

# 임시 파일로 저장
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
    tmp_file.write(uploaded_file.read())
    video_path = tmp_file.name

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error("영상을 읽을 수 없습니다. 다른 파일을 시도해 주세요.")
    os.unlink(video_path)
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

st.success(f"영상 로드 완료 | {width}×{height} | {fps:.1f} FPS | {duration:.1f}초")

# 구간 선택
st.subheader("📍 분석 구간 선택")
col1, col2 = st.columns(2)
with col1:
    start_sec = st.slider("시작 시간 (초)", 0.0, duration, 0.0, 0.1)
with col2:
    end_sec = st.slider("종료 시간 (초)", 0.0, duration, min(8.0, duration), 0.1)

if start_sec >= end_sec:
    st.error("종료 시간이 시작 시간보다 커야 합니다.")
    st.stop()

st.caption(f"선택 구간 길이: **{end_sec - start_sec:.2f}초**")

# 분석 실행
if st.button("🔍 선택 구간 분석 시작", type="primary", use_container_width=True):
    with st.spinner("영상을 분석하고 있습니다..."):
        positions = process_segment(video_path, start_sec, end_sec, fps, sample_step=2)
    
    if not positions:
        st.warning("선택한 구간에서 공을 검출하지 못했습니다. 다른 구간을 시도해 보세요.")
    else:
        st.success(f"분석 완료! 총 {len(positions)}개의 공 위치를 검출했습니다.")
        
        # 결과 저장
        st.session_state['positions'] = positions
        st.session_state['video_path'] = video_path
        st.session_state['start_sec'] = start_sec
        st.session_state['end_sec'] = end_sec
        st.session_state['fps'] = fps

# 분석 결과 표시
if 'positions' in st.session_state:
    positions = st.session_state['positions']
    
    st.subheader("📊 분석 결과")
    
    tab1, tab2, tab3 = st.tabs(["Top-down 뷰", "3D 궤적", "원본 영상"])
    
    with tab1:
        st.markdown("**테이블 위에서 본 공의 움직임**")
        fig_top = create_topdown_plot(positions)
        st.pyplot(fig_top, use_container_width=True)
        st.caption("색상이 변하는 부분이 시간의 흐름을 나타냅니다. (보라색 → 노란색)")
    
    with tab2:
        st.markdown("**3D 공간에서 본 공 궤적** (마우스로 회전 가능)")
        fig_3d = create_3d_plot(positions)
        st.pyplot(fig_3d, use_container_width=True)
        st.caption("※ 현재 버전은 높이(z)를 0으로 가정합니다. 실제 높이 분석은 추후 업데이트 예정")
    
    with tab3:
        st.markdown("**원본 영상 재생**")
        st.video(uploaded_file, start_time=st.session_state['start_sec'])
        st.caption("브라우저의 재생 속도 조절 기능을 사용하세요 (보통 0.25x ~ 0.5x 권장)")

    # 추가 정보
    with st.expander("💡 분석 팁 및 제한사항"):
        st.markdown("""
        - **스마트폰에서 보기 좋게** 하려면 가로 모드로 전환하세요.
        - 공 검출 정확도는 조명과 배경에 따라 달라질 수 있습니다.
        - 더 정확한 분석을 원하시면 **테이블 보정** 기능이 포함된 데스크톱 버전을 사용해 주세요.
        - 긴 영상(1분 이상)은 분석 시간이 오래 걸릴 수 있습니다.
        """)

# 푸터
st.markdown("---")
st.caption("탁구 3D 분석 웹앱 v1.1 | Streamlit 기반 | 데스크톱 및 모바일 브라우저 지원")
