import tempfile

import cv2
import streamlit as st
import pandas as pd
import numpy as np
# def infer_uploaded_video(conf, model):
#     """
#     Execute inference for uploaded video
#     :param conf: Confidence of YOLOv8 model
#     :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#     :return: None
#     """
#
#
#     if source_video:
#         if st.button("Execution"):
#             with st.spinner("Running..."):
#                 try:
#                     tfile = tempfile.NamedTemporaryFile()
#                     tfile.write(source_video.read())
#                     vid_cap = cv2.VideoCapture(
#                         tfile.name)
#                     st_frame = st.empty()
#                     while (vid_cap.isOpened()):
#                         success, image = vid_cap.read()
#                         if success:
#                             _display_detected_frames(conf,
#                                                      model,
#                                                      st_frame,
#                                                      image
#                                                      )
#                         else:
#                             vid_cap.release()
#                             break
#                 except Exception as e:
#                     st.error(f"Error loading video: {e}")


st.sidebar.header("DL Model Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ["Video", "Webcam"])

source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )


if source_video:
    with open(f"upload/{source_video.name}","wb") as f:
        f.write(source_video.read())
    st.video(source_video)
    # tfile = tempfile.NamedTemporaryFile()
    # tfile.write(source_video.read())
    vid_cap = cv2.VideoCapture(f"upload/{source_video.name}")
    st_frame = st.empty()
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            st_frame.image(image,
                           caption='Detected Video',
                           channels="BGR",
                           use_column_width=True,
                           output_format="JPEG")
        else:
            break