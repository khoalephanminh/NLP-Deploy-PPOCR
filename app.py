import codecs
import json
import logging
import sys
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
        "--rec_image_shape", "3, 32, 320",
        "--draw_img_save_dir", draw_img_save_dir,
        "--rec_char_dict_path", "./ppocr/utils/chinese_cht_dict.txt",
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

atexit.register(cleanup_temp_dir)

from PIL import Image

def main():
    # Upload image
    st.sidebar.write("""
    # Chinese PaddleOCR Streamlit Demo
    
    ---
    """)
    uploaded_files = st.sidebar.file_uploader('Please select an image', type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)
    print('uploaded_file:', uploaded_files)

    col1, col2 = st.columns(2)

    option_task = st.sidebar.radio('Please select a task to perform', ('View Original Image', 'Text Detection', 'Text Recognition', 'End to End OCR'))

    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            # Convert format
            img = bytes_to_numpy(bytes_data, channels='RGB')
            pil_img = Image.open(uploaded_file)

            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            # Save the uploaded file to a temporary location
            temp_image_path = os.path.join(temp_dir, uploaded_file.name)

            os.makedirs('data', exist_ok=True)
            os.makedirs('data/img', exist_ok=True)
            with open(temp_image_path, "wb") as f:
                f.write(bytes_data)

            use_gpu = False

            if option_task == 'View Original Image':
                with col1:
                    st.image(img, caption='Original Image')

            elif option_task in ['Text Detection', 'Text Recognition', 'End to End OCR']:
                if option_task == 'Text Detection':
                    result_image_path = os.path.join("./inference_results", f"det_res_{uploaded_file.name}")
                    # if os.path.exists(result_image_path):
                    #     st.image(result_image_path, caption='Detection Result Image')
                    #     pass

                    # Call the predict_det.py script
                    det_model_dir = "output/det_finetune"
                    stdout, stderr = call_predict_det(temp_image_path, det_model_dir, use_gpu)

                    det_results_path = os.path.join("./inference_results", "det_results.txt")
                    # Display the result image
                    if os.path.exists(result_image_path):
                        with open(det_results_path, "r", encoding='utf-8') as file:
                            det_results_content = file.read()
                            json_content = det_results_content.split('\t', 1)[1]
                            results = json.loads(json_content)

                            with col2:
                                saved_json = {
                                    'num_boxes': len(results),
                                    'height': pil_img.height,
                                    'width': pil_img.width,
                                    'patches': []
                                }

                                with ZipFile('data/patches.zip', 'w') as zip_file:
                                    with open(f'data/data.json', 'w', encoding='utf-8') as json_file:
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

                                        worksheet.write_row(0, 0, ['Image_Name', 'ID', 'Image Box'], header_format)

                                        for idx, points in enumerate(results):
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
                                            if width > max_size[0] or height > max_size[1]:
                                                image_patch.thumbnail(max_size, Image.LANCZOS)

                                            # Image Name
                                            image_name = uploaded_file.name

                                            # Image ID
                                            image_id = image_name.split('.')[0] + '.' + str(idx + 1).zfill(3)

                                            with st.expander(f':red[**Bounding box {idx + 1:02d}**:] {str(points)}'):
                                                col21, col22 = st.columns([1, 7])
                                                with col21:
                                                    st.image(image_patch)
                                                with col22:
                                                    saved_json['patches'].append({
                                                        'image_name': image_name,
                                                        'image_id': image_id,
                                                        'points': points,
                                                        'height': height,
                                                        'width': width
                                                    })

                                                    data = {
                                                        "Key": ["Image ID", "Points", "Height", "Width"],
                                                        "Value": [image_id, points, height, width]
                                                    }
                                                    df = pd.DataFrame(data)
                                                    st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

                                            # Save the image patch, it should be saved in the zip file
                                            image_patch_path = f'data/img/{image_id}.jpg'
                                            image_patch.save(image_patch_path)
                                            zip_file.write(image_patch_path)
                                            # Save the patch to csv
                                            worksheet.write(idx + 1, 0, image_name, normal_format)
                                            worksheet.write(idx + 1, 1, image_id, normal_format)
                                            worksheet.write(idx + 1, 2, json.dumps(points), normal_format)

                                        json.dump(saved_json, json_file, ensure_ascii=False, indent=4)

                                        column_width = [20, 20, 60]
                                        for col, width in enumerate(column_width):
                                            worksheet.set_column(col, col, width)

                                        workbook.close()

                                    zip_file.write('data/data.json')
                                    zip_file.write('data/data.xlsx')


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

                                # Add a download button
                                st.download_button(
                                    label=f'📥 Export Raw Detection Results: det_results.txt',
                                    data=det_results_content,
                                    file_name="det_results.txt",
                                    mime="text/plain",
                                    use_container_width=True,
                                )

                                st.download_button(
                                    label=f'🖼️ Download patches',
                                    data=open('data/patches.zip', 'rb'),
                                    file_name='patches.zip',
                                    use_container_width=True,
                                )

                                st.image(result_image_path, caption='Detection Result Image')

                elif option_task == 'Text Recognition':
                    # Call the predict_rec.py script
                    rec_model_dir = "output/rec_finetune"
                    rec_image_shape = "3, 32, 320"
                    rec_char_dict_path = "./ppocr/utils/chinese_cht_dict.txt"
                    stdout, stderr = call_predict_rec(temp_image_path, rec_model_dir, rec_image_shape, rec_char_dict_path, use_gpu)

                elif option_task == 'End to End OCR':
                    draw_img_save_dir = "./e2e_visualize/"
                    result_image_path = os.path.join(draw_img_save_dir, uploaded_file.name)
                    # if os.path.exists(result_image_path):
                    #     st.image(result_image_path, caption='Result Image')
                    #     pass

                    # Call the predict_system.py script
                    det_model_dir = "output/det_finetune"
                    rec_model_dir = "output/rec_finetune"
                    stdout, stderr = call_predict_system(det_model_dir, rec_model_dir, temp_dir, draw_img_save_dir, use_gpu)

                    system_results_path = os.path.join("./e2e_visualize", "system_results.txt")
                    # Display the result image
                    if os.path.exists(result_image_path):
                        with open(system_results_path, "r", encoding='utf-8') as file:
                            system_results_content = file.read()
                            json_content = system_results_content.split('\t', 1)[1]
                            results = json.loads(json_content)

                            with col2:
                                saved_json = {
                                    'num_boxes': len(results),
                                    'height': pil_img.height,
                                    'width': pil_img.width,
                                    'patches': []
                                }

                                with ZipFile('data/patches.zip', 'w') as zip_file:
                                    with open(f'data/data.json', 'w', encoding='utf-8') as json_file:
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

                                        for idx, result in enumerate(results):
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
                                            if width > max_size[0] or height > max_size[1]:
                                                image_patch.thumbnail(max_size, Image.LANCZOS)

                                            # Image Name
                                            image_name = uploaded_file.name

                                            # Image ID
                                            image_id = image_name.split('.')[0] + '.' + str(idx + 1).zfill(3)



                                            with st.expander(f':red[**Text {idx + 1:02d}**:] {transcription}'):
                                                col21, col22 = st.columns([1, 7])
                                                with col21:
                                                    st.image(image_patch)
                                                with col22:
                                                    saved_json['patches'].append({
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
                                                    df = pd.DataFrame(data)
                                                    st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

                                            # Save the image patch, it should be saved in the zip file
                                            image_patch_path = f'data/img/{image_id}.jpg'
                                            image_patch.save(image_patch_path)
                                            zip_file.write(image_patch_path)
                                            # Save the patch to csv
                                            worksheet.write(idx + 1, 0, image_name, normal_format)
                                            worksheet.write(idx + 1, 1, image_id, normal_format)
                                            worksheet.write(idx + 1, 2, json.dumps(points), normal_format)
                                            worksheet.write(idx + 1, 3, transcription)

                                        json.dump(saved_json, json_file, ensure_ascii=False, indent=4)

                                        column_width = [20, 20, 60, 100]
                                        for col, width in enumerate(column_width):
                                            worksheet.set_column(col, col, width)

                                        workbook.close()

                                    zip_file.write('data/data.json')
                                    zip_file.write('data/data.xlsx')

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
                                    data=system_results_content,
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

                                st.image(result_image_path, caption='Result Image')

                # Display the output
                with col1:
                    with st.expander('Standard Output:'):
                        st.text(stdout)
                    with st.expander("Standard Error:"):
                        st.text(stderr)
main()