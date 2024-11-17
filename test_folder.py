import os
import shutil

# 이미지 파일을 정리할 폴더 경로를 지정하세요
source_folder = '/Users/choizeus/Pictures/Photo/train/model'

# 이미지 파일 확장자 목록
image_extensions = ['.png', '.jpg', '.jpeg']

# 폴더 내 파일 확인
for filename in os.listdir(source_folder):
    # 파일 확장자 체크
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        # 확장자 바로 앞의 숫자 부분 추출
        base, ext = os.path.splitext(filename)
        if '_' in base and base.split('_')[-1].isdigit():
            target_folder = base.split('_')[-1]  # 숫자 부분 추출

            # 이동할 폴더 생성
            target_path = os.path.join(source_folder, target_folder)
            os.makedirs(target_path, exist_ok=True)

            # 파일 이동
            shutil.move(os.path.join(source_folder, filename), os.path.join(target_path, filename))
            print(f"Moved {filename} to folder {target_folder}")

print("모든 파일 이동 완료")
