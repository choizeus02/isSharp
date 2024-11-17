
# 학습에 사용하지 않게 하기 위해 라벨링을 지우는 작업

import os


def remove_suffix_from_images(folder_path):
    # 지정한 폴더를 순회
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 이미지 파일만 처리 (.jpg, .jpeg, .png 확장자)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 파일명과 확장자 분리
                base_name, ext = os.path.splitext(file)

                # 파일명이 '_0', '_1', '_2' 등으로 끝나는지 확인
                if base_name.endswith(('_0', '_1', '_2')):
                    # 접미사 제거
                    new_base_name = base_name.rsplit('_', 1)[0]
                    new_filename = f"{new_base_name}{ext}"

                    # 파일의 전체 경로
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, new_filename)

                    # 파일 이름 변경
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed '{file}' to '{new_filename}'")
                    except Exception as e:
                        print(f"Failed to rename '{file}': {e}")


# 사용 예시
remove_suffix_from_images('/Users/choizeus/Pictures/Photo/FUJI/2024/무제 폴더')
