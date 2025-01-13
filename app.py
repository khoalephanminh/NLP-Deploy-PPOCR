import codecs
import json
import logging
import sys
from numpy import std
from ppocr.utils.utility import check_and_read, get_image_file_list
import streamlit as st
import subprocess
import os
import shutil
import atexit
from utils import bytes_to_numpy
from style import custom_css
import pandas as pd
from zipfile import ZipFile
from xlsxwriter.workbook import Workbook
import cv2

st.set_page_config('Chinese PaddleOCR for server - Group 02', layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

# Ensure UTF-8 environment for the subprocess
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

def call_predict_system(det_model_dir, rec_model_dir, image_path, draw_img_save_dir, use_gpu):
    cmd = [
        sys.executable, "tools/infer/predict_system.py",
        "--det_model_dir", det_model_dir,
        "--rec_model_dir", rec_model_dir,
        "--image_dir", image_path,
        "--rec_image_shape", "3, 32, 640",
        "--draw_img_save_dir", draw_img_save_dir,
        "--rec_char_dict_path", "./ppocr/utils/chinese_cht_dict_update.txt",
        "--det_db_unclip_ratio=2.5",
        "--rec_batch_num=1",
        "--use_gpu", str(use_gpu).lower()
    ]
    result = subprocess.run(cmd, capture_output=True, env=env)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")

    return stdout, stderr

def call_predict_det(image_path, det_model_dir, use_gpu):
    cmd = [
        sys.executable, "tools/infer/predict_det.py",
        "--image_dir", image_path,
        "--det_model_dir", det_model_dir,
        "--use_gpu", str(use_gpu).lower()
    ]
    result = subprocess.run(cmd, capture_output=True, env=env)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")

    return stdout, stderr

def call_predict_rec(image_path, rec_model_dir, rec_image_shape, rec_char_dict_path, use_gpu):
    cmd = [
        sys.executable, "tools/infer/predict_rec.py",
        "--image_dir", image_path,
        "--rec_model_dir", rec_model_dir,
        "--rec_image_shape", rec_image_shape,
        "--rec_char_dict_path", rec_char_dict_path,
        "--use_gpu", str(use_gpu).lower()
    ]
    
    result = subprocess.run(cmd, capture_output=True, env=env)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    
    return stdout, stderr

def cleanup_temp_dir():
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Deleted {temp_dir} directory")

    data_dir = "data"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Deleted {data_dir} directory")

    data_dir = "e2e_visualize"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Deleted {data_dir} directory")

    data_dir = "inference_results"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Deleted {data_dir} directory")

atexit.register(cleanup_temp_dir)

from PIL import Image

def preprocess(uploaded_files):
    cleanup_temp_dir()
    st.session_state.data = {}
    image = {}
    is_rec = False
    is_pdf = False

    image_names = []

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/img', exist_ok=True)


    if uploaded_files[0].type.startswith("application/pdf"):
        is_pdf = True
        bytes_data = uploaded_files[0].getvalue()
        temp_image_path = os.path.join(temp_dir, uploaded_files[0].name)
        with open(temp_image_path, "wb") as f:
            f.write(bytes_data)

        imgs, flag_gif, flag_pdf = check_and_read(temp_image_path)

        for idx, img in enumerate(imgs):
            # Convert back from OpenCV format to PIL image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image[f'{uploaded_files[0].name}_{idx}.png'] = {
                'img': img,
                'pil_img': pil_img
            }
            image_names.append(f'{uploaded_files[0].name}_{idx}.png')


    if uploaded_files[0].type.startswith("image"):
        for uploaded_file in uploaded_files:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            img = bytes_to_numpy(bytes_data, channels='RGB')
            pil_img = Image.open(uploaded_file)

            if pil_img.height < 100: 
                is_rec = True

            image[uploaded_file.name] = {
                'img': img,
                'pil_img': pil_img
            }

            temp_image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_image_path, "wb") as f:
                f.write(bytes_data)

            image_names.append(uploaded_file.name)

    # Call the predict_det.py script
    use_gpu = False

    draw_img_save_dir = "./e2e_visualize/"
    # if os.path.exists(result_image_path):
    #     st.image(result_image_path, caption='Result Image')
    #     pass

    # Call the predict_system.py script
    det_model_dir = "model/det_finetune"
    rec_model_dir = "model/rec_finetune"

    if is_rec:
        stdout, stderr = call_predict_rec(temp_dir, rec_model_dir, '3, 32, 640', "./ppocr/utils/chinese_cht_dict_update.txt", use_gpu)
    else:
        stdout, stderr = call_predict_system(det_model_dir, rec_model_dir, temp_dir, draw_img_save_dir, use_gpu)
        # if is_pdf: os.remove(temp_image_path)

    system_results_path = os.path.join("./e2e_visualize", "system_results.txt")

    if is_rec: return stdout, stderr, is_rec, image_names

    with open(system_results_path, "r", encoding='utf-8') as file:
        lines = file.readlines()
        with ZipFile('data/patches.zip', 'w') as zip_file:
            with open(f'data/data.json', 'w', encoding='utf-8') as json_file:
                saved_json = []
                row_idx = 0
                workbook = Workbook('data/data.xlsx')
                workbook.formats[0].set_font_size(14)
                workbook.formats[0].set_font_name("Nom Na Tong")

                normal_format = workbook.add_format({"font_color": "black"})

                header_format = workbook.add_format(
                    {
                        "bold": True,
                        "align": "center",
                        "valign": "vcenter",
                        "bg_color": "#F0F8FF",
                    }
                )

                worksheet = workbook.add_worksheet()

                worksheet.write_row(0, 0, ['Image_Name', 'ID', 'Image Box', 'OCR Text'], header_format)
                for system_results_content in lines:
                    file_name = system_results_content.split('\t', 1)[0]
                    if not '.png' in file_name and not '.jpg' in file_name and not '.jpeg' in file_name:
                        file_name = file_name + '.png'

                    json_content = system_results_content.split('\t', 1)[1]
                    results = json.loads(json_content)

                    saved_json.append({
                        'image_name': file_name,
                        'num_boxes': len(results),
                        'height': pil_img.height,
                        'width': pil_img.width,
                        'patches': []
                    })

                    pil_img = image[file_name]['pil_img']

                    st.session_state.data[file_name] = {
                        'current_image_path': f'./e2e_visualize/{file_name.replace(".pdf", "")}',   
                        'boxes': []
                    }

                    for idx, result in enumerate(results):
                        row_idx += 1

                        transcription = result["transcription"]
                        points = result["points"]
                        x_min = min(point[0] for point in points)
                        y_min = min(point[1] for point in points)
                        x_max = max(point[0] for point in points)
                        y_max = max(point[1] for point in points)
                        width = x_max - x_min
                        height = y_max - y_min

                        # Crop the image patch
                        image_patch = pil_img.crop((x_min, y_min, x_max, y_max))

                        # Resize the image patch if it is too large
                        max_size = (300, 300)
                        # if width > max_size[0] or height > max_size[1]:
                        #     image_patch.thumbnail(max_size, Image.LANCZOS)

                        # Image Name
                        image_name = file_name

                        # Image ID
                        image_id = image_name.split('.')[0] + '.' + str(idx + 1).zfill(3)

                        saved_json[-1]['patches'].append({
                            'image_name': image_name,
                            'image_id': image_id,
                            'transcription': transcription,
                            'points': points,
                            'height': height,
                            'width': width
                        })

                        data = {
                            "Key": ["Image ID", "Chinese", "Points", "Height", "Width"],
                            "Value": [image_id, transcription, points, height, width]
                        }

                        st.session_state.data[file_name]['boxes'].append({
                            'image_patch': image_patch,
                            'data': data,
                            'transcription': transcription
                        })

                        # Save the image patch, it should be saved in the zip file
                        image_patch_path = f'data/img/{image_id}.jpg'
                        image_patch.save(image_patch_path)
                        zip_file.write(image_patch_path)
                        # Save the patch to csv
                        worksheet.write(row_idx, 0, image_name, normal_format)
                        worksheet.write(row_idx, 1, image_id, normal_format)
                        worksheet.write(row_idx, 2, json.dumps(points), normal_format)
                        worksheet.write(row_idx, 3, transcription)

                json.dump(saved_json, json_file, ensure_ascii=False, indent=4)

                column_width = [20, 20, 60, 100]
                for col, width in enumerate(column_width):
                    worksheet.set_column(col, col, width)

                workbook.close()

            zip_file.write('data/data.json')
            zip_file.write('data/data.xlsx')

    return stdout, stderr, is_rec, image_names 

def main():
    # Upload image
    st.sidebar.write("""
    # Chinese PaddleOCR Streamlit Demo
    
    ---
    """)
    uploaded_files = st.sidebar.file_uploader('Please select an image', type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)

    # Initialize session state attributes if they don't exist
    if 'stdout' not in st.session_state:
        st.session_state.stdout = ''
    if 'stderr' not in st.session_state:
        st.session_state.stderr = ''
    if 'is_rec' not in st.session_state:
        st.session_state.is_rec = False
    if 'image_names' not in st.session_state:
        st.session_state.image_names = []

    if len(uploaded_files) > 0:
        if 'data' not in st.session_state:
            stdout, stderr, is_rec, image_names = preprocess(uploaded_files)
            st.session_state.current_index = 0
            st.session_state.stdout = stdout
            st.session_state.stderr = stderr
            st.session_state.is_rec = is_rec
            st.session_state.image_names = image_names

    stdout = st.session_state.stdout
    stderr = st.session_state.stderr
    is_rec = st.session_state.is_rec
    image_names = st.session_state.image_names

    if 'data' in st.session_state:
        col1, col2 = st.columns(2)
        
        # Display the output
        with col1:
            with st.expander('Standard Output:'):
                st.text(stdout)
            with st.expander("Standard Error:"):
                st.text(stderr)

        if is_rec: return

        # current_data = st.session_state.data[uploaded_files[st.session_state.current_index].name]
        current_data = st.session_state.data[image_names[st.session_state.current_index]]


        with col2:
            for idx, box in enumerate(current_data['boxes']):
                with st.expander(f':red[**Text {idx + 1:02d}**:] {box["transcription"]}'):
                    col21, col22 = st.columns([1, 7])
                    with col21:
                        st.image(box['image_patch'])
                    with col22:
                        df = pd.DataFrame(box['data'])
                        st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

        with col1:
            st.download_button(
                label=f'📥 Export OCR results (XSLX): data.xlsx',
                data=open('data/data.xlsx', 'rb'),
                file_name=f'data.xlsx',
                use_container_width=True,
            )

            st.download_button(
                label=f'📥 Export OCR results (JSON): data.json',
                data=open('data/data.json', 'rb'),
                file_name=f'data.json',
                use_container_width=True,
            )

            st.download_button(
                label=f'📥 Export raw OCR Results: e2e_results.txt',
                data=open('e2e_visualize/system_results.txt', 'rb'),
                file_name="e2e_results.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.download_button(
                label=f'🖼️ Download patches',
                data=open('data/patches.zip', 'rb'),
                file_name='patches.zip',
                use_container_width=True,
            )

            col11, col12 = st.columns(2)
            with col11:
                if st.session_state.current_index > 0:
                    if st.button('Previous Image'):
                        st.session_state.current_index -= 1
                        st.rerun()
                else:
                    st.write('')
            with col12:
                if st.session_state.current_index < len(image_names) - 1:
                    if st.button('Next Image'):
                        st.session_state.current_index += 1
                        st.rerun()
                else:
                    st.write('')
            st.image(current_data['current_image_path'], caption=os.path.basename(current_data['current_image_path']))

main()