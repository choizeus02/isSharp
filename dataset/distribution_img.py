

# 많은 이미지를 포함한 폴더 나누기/합치기
# 하위에 1,2,3,4... 폴더 생성하여 이미지 100개 씩 분배
# Image_rename.py 실행시 메모리 터짐으로 인한 해결책


import os
import shutil


def organize_files_in_batches(folder_path, batch_size=100):
    # 폴더 내 파일 목록 가져오기
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_files = len(files)

    # 배치별로 파일을 정리
    for i in range(0, total_files, batch_size):
        # 새로운 폴더 이름 생성 (1, 2, 3, ...)
        batch_folder_name = str(i // batch_size + 1)
        batch_folder_path = os.path.join(folder_path, batch_folder_name)

        # 새로운 폴더 생성
        os.makedirs(batch_folder_path, exist_ok=True)

        # 배치 내의 파일들을 새로운 폴더로 이동
        for file in files[i:i + batch_size]:
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(batch_folder_path, file)
            shutil.move(old_path, new_path)
            print(f"Moved '{file}' to '{batch_folder_name}'")



def undo_organize_files_in_batches(folder_path):
    # 폴더 내 하위 폴더 목록 가져오기
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)

        # 하위 폴더 내의 모든 파일을 상위 폴더로 이동
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            if os.path.isfile(file_path):
                new_path = os.path.join(folder_path, file)
                shutil.move(file_path, new_path)
                print(f"Moved '{file}' back to '{folder_path}'")

        # 하위 폴더 삭제
        os.rmdir(subfolder_path)
        print(f"Removed folder '{subfolder_path}'")


# 사용 예시
undo_organize_files_in_batches('/Users/choizeus/Pictures/Photo/train/FUJI/2024/2:29 (도쿄)/1')

# 사용 예시
# organize_files_in_batches('/Users/choizeus/Pictures/Photo/FUJI/2024/2:29 (도쿄)/1', batch_size=100)
