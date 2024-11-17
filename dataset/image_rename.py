
# 이미지 이름 변경기
# 분류기(classifiers.py)로 이미지 라벨링을 진행 후 사용자가 하나하나 확인 후 검수 하는 py


import os
import cv2
from PIL import Image, ImageTk
import tkinter as tk
def display_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        cv2.imshow("Image", image)
        cv2.waitKey(1)  # 창을 표시
        cv2.destroyAllWindows()
        del image
    else:
        print(f"Failed to load {image_path}")


def display_image1(image_path, width=400, height=400):
    # tkinter 윈도우 생성
    root = tk.Tk()
    root.title("Image")

    # 이미지를 Pillow로 열고 리사이즈
    img = Image.open(image_path)
    img = img.resize((width, height), Image.ANTIALIAS)  # 400x400 픽셀로 리사이즈
    tk_image = ImageTk.PhotoImage(img)

    # 레이블에 이미지 추가 및 표시
    label = tk.Label(root, image=tk_image)
    label.pack()

    # 1초 후 자동으로 창 닫기 (필요시 조정 가능)
    root.after(1000, root.destroy)
    root.mainloop()

    img.close()  # Pillow 이미지 해제

def rename_image_with_suffix(image_path, suffix):
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    if base_name.endswith(('_0', '_1', '_2', '_unknown')):
        base_name = base_name.rsplit('_', 1)[0]

    new_filename = f"{base_name}{suffix}{ext}"
    new_path = os.path.join(os.path.dirname(image_path), new_filename)

    try:
        os.rename(image_path, new_path)
        print(f"Renamed {image_path} to {new_filename}")
    except PermissionError:
        print(f"Skipping {image_path}: Permission denied.")

def classify_and_rename_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)

                # 이미지 보여주기
                # display_image1(image_path, width=400, height=400)
                display_image(image_path)

                # 사용자 입력 받기
                suffix_map = {'0': '_0', '1': '_1', '2': '_2'}
                user_input = input("Enter 0 for Blurry, 1 for Sharp, or 2 for Shallow Focus: ")

                # 입력을 받으면 이미지 창 닫기

                # 입력이 유효한 경우 이름 변경
                if user_input in suffix_map:
                    rename_image_with_suffix(image_path, suffix_map[user_input])
                else:
                    print("Invalid input. Skipping this image.")

# 사용 예시
classify_and_rename_images('/Users/choizeus/Pictures/Photo/FUJI/2024/2:29 (도쿄)/1/8')
