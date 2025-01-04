import codecs
import logging
import sys
import streamlit as st
import subprocess
import os
import shutil
import atexit
from utils import bytes_to_numpy

    
st.write("""
# PaddleOCR Streamlit Demo

---
""")

# Ensure UTF-8 environment for the subprocess
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

def call_predict_system(det_model_dir, rec_model_dir, image_path, draw_img_save_dir, use_gpu):
    cmd = [
        sys.executable, "tools/infer/predict_system.py",
        "--det_model_dir", det_model_dir,
        "--rec_model_dir", rec_model_dir,
        "--image_dir", image_path,
        "--draw_img_save_dir", draw_img_save_dir,
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

atexit.register(cleanup_temp_dir)

def main():
    # Upload image
    uploaded_file = st.sidebar.file_uploader('Please select an image', type=['png', 'jpg', 'jpeg'])
    print('uploaded_file:', uploaded_file)

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # Convert format
        img = bytes_to_numpy(bytes_data, channels='RGB')

        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        # Save the uploaded file to a temporary location
        temp_image_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_image_path, "wb") as f:
            f.write(bytes_data)

        option_task = st.sidebar.radio('Please select a task to perform', ('View Original Image', 'Text Detection', 'Text Recognition', 'End to End OCR'))
        use_gpu = False

        if option_task == 'View Original Image':
            st.image(img, caption='Original Image')

        elif option_task in ['Text Detection', 'Text Recognition', 'End to End OCR']:
            if option_task == 'Text Detection':
                result_image_path = os.path.join("./inference_results", f"det_res_{uploaded_file.name}")
                # if os.path.exists(result_image_path):
                #     st.image(result_image_path, caption='Detection Result Image')
                #     pass

                # Call the predict_det.py script
                det_model_dir = "./output/det_pretrain"
                stdout, stderr = call_predict_det(temp_image_path, det_model_dir, use_gpu)

                # Display the result image
                if os.path.exists(result_image_path):
                    st.image(result_image_path, caption='Detection Result Image')

            elif option_task == 'Text Recognition':
                # Call the predict_rec.py script
                rec_model_dir = "./output/rec_pretrain/Student"
                rec_image_shape = "3, 48, 320"
                rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt"
                stdout, stderr = call_predict_rec(temp_image_path, rec_model_dir, rec_image_shape, rec_char_dict_path, use_gpu)

            elif option_task == 'End to End OCR':
                draw_img_save_dir = "./e2e_visualize/"
                result_image_path = os.path.join(draw_img_save_dir, uploaded_file.name)
                # if os.path.exists(result_image_path):
                #     st.image(result_image_path, caption='Result Image')
                #     pass

                # Call the predict_system.py script
                det_model_dir = "./output/det_pretrain"
                rec_model_dir = "./output/rec_pretrain/Student"
                stdout, stderr = call_predict_system(det_model_dir, rec_model_dir, temp_image_path, draw_img_save_dir, use_gpu)

                # Display the result image
                if os.path.exists(result_image_path):
                    st.image(result_image_path, caption='Result Image')

            # Display the output
            st.text("Standard Output:")
            st.text(stdout)
            st.text("Standard Error:")
            st.text(stderr)

main()