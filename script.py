import os
import json
import tempfile
import streamlit as st
import cv2


def process_videos(file_names, output_video_file, fourcc, fps, frame_size, pose):
    for file_name in file_names:
        up_file = open(file_name, 'rb')
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(up_file.read())
        tfile.close()
        
        video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)
        
        txt = st.sidebar.markdown(ip_vid_str, unsafe_allow_html=True)
        ip_video = st.sidebar.video(tfile.name)
        
        inward_arr = get_inward_data(up_file.name)
        outward_arr = get_outward_data(up_file.name)
        
        vf = cv2.VideoCapture(tfile.name)
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            timestamp = vf.get(cv2.CAP_PROP_POS_MSEC)
            inward_knee = (inward_arr[0] < timestamp / 1000 < inward_arr[1])
            outward_knee = (outward_arr[0] < timestamp / 1000 < outward_arr[1])
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_frame, _ = upload_process_frame.process(frame, pose, inward_knee, outward_knee)
            
            video_output.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))
        
        angles_json = {
            "len": len(angles_array),
            "arr": angles_array,
            "id": up_file.name
        }
        
        json_file_path = './angles.json'
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
        else:
            data = []
        
        data.append(angles_json)
        
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)
        
        vf.release()
        video_output.release()
        stframe.empty()
        ip_video.empty()
        txt.empty()
        tfile.close()
        up_file.close()

with open('file_names.txt', 'r') as f:
    file_names = [line.strip() for line in f.readlines()]

output_video_file = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
frame_size = (640, 480)
pose = None  # Replace with actual pose object

process_videos(file_names, output_video_file, fourcc, fps, frame_size, pose)