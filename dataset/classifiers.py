
# 이미지 분류기
# 폴더를 순회하며 정해진 기준에 맞춰 이미지 라벨링
# 이미지명 뒤에 라벨링을 추가

import os
import cv2
import numpy as np

# 이미지 흐림 정도 측정(라플라시안 변동성)
def measure_blurriness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance

# 모션 블러 감지
def detect_motion_blur(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel_h = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_v = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    horizontal_edges = cv2.filter2D(image, -1, kernel_h)
    vertical_edges = cv2.filter2D(image, -1, kernel_v)
    total_edges = np.sum(horizontal_edges) + np.sum(vertical_edges)
    if np.sum(horizontal_edges) > np.sum(vertical_edges):
        direction = "Vertical"
    elif np.sum(vertical_edges) > np.sum(horizontal_edges):
        direction = "Horizontal"
    else:
        direction = "None"
    return total_edges, direction

# 심도 흐림 감지
def detect_depth_blur(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    h_step, w_step = h // 8, w // 8

    regions = [
        image[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
        for i in range(8) for j in range(8)
    ]
    blur_values = [round(cv2.Laplacian(region, cv2.CV_64F).var(), 1) for region in regions]
    blur_values_8x8 = [blur_values[i * 8:(i + 1) * 8] for i in range(8)]

    blur_mean = np.mean(blur_values)
    blur_variance = np.var(blur_values)
    max_min_diff = max(blur_values) - min(blur_values)
    max_min_to_mean_ratio = max_min_diff / blur_mean if blur_mean != 0 else 0

    top_20_percent = np.percentile(blur_values, 95)
    bottom_20_percent = np.percentile(blur_values, 20)
    top_bottom_diff = top_20_percent - bottom_20_percent
    max_min_ratio = max(blur_values) / min(blur_values) if min(blur_values) != 0 else 0
    top_bottom_ratio = top_20_percent / bottom_20_percent if bottom_20_percent != 0 else 0
    variance_to_mean_ratio = blur_variance / blur_mean if blur_mean != 0 else 0
    high_blur_count = sum(1 for value in blur_values if value > blur_mean)

    depth_blur_detected = high_blur_count >= 10 and (variance_to_mean_ratio > 0.5)

    return (blur_values_8x8, blur_mean, blur_variance, high_blur_count, variance_to_mean_ratio,
            max_min_diff, top_bottom_diff, max_min_to_mean_ratio, max_min_ratio, top_bottom_ratio)

# 가중치 기반 분류 함수
def classify_image_weighted(blurriness, motion_blur_score, blur_mean, blur_variance,
                            variance_to_mean_ratio, max_min_diff, max_min_ratio,
                            top_bottom_diff, max_min_to_mean_ratio, top_bottom_ratio):
    weights = {
        'blurriness': 100,
        'motion_blur_score': 1,
        'blur_mean': 2,
        'blur_variance': 100,
        'variance_to_mean_ratio': 15,
        'max_min_diff': 5,
        'max_min_ratio': 3,
        'top_bottom_diff': 100,
        'max_min_to_mean_ratio': 12,
        'top_bottom_ratio': 5
    }

    # 각 클래스의 평균 값 정의
    blurry_mean = [34.3, 193139676.66, 35.29, 137.15, 4.68, 91.78, 6.82, 42.35, 2.16, 2.56]
    shallow_focus_mean = [690.53, 45515273.89, 694.85, 161561.01, 561.45, 2800.52, 654.97, 1569.5, 6.62, 65.8]
    sharp_mean = [3435.84, 91958391.49, 3454.87, 5446786.30, 2208.53, 12261.04, 2292.73, 7787.95, 4.31, 156.17]

    # 새로운 이미지의 특징 벡터
    image_features = [blurriness, motion_blur_score, blur_mean, blur_variance,
                      variance_to_mean_ratio, max_min_diff, max_min_ratio,
                      top_bottom_diff, max_min_to_mean_ratio, top_bottom_ratio]

    # 가중치를 적용한 거리 계산
    blurry_dist = np.sqrt(sum(weights[key] * (image_features[i] - blurry_mean[i]) ** 2
                              for i, key in enumerate(weights)))
    shallow_focus_dist = np.sqrt(sum(weights[key] * (image_features[i] - shallow_focus_mean[i]) ** 2
                                     for i, key in enumerate(weights)))
    sharp_dist = np.sqrt(sum(weights[key] * (image_features[i] - sharp_mean[i]) ** 2
                             for i, key in enumerate(weights)))

    # 최소 거리로 분류
    distances = {'Blurry': blurry_dist, 'Shallow Focus': shallow_focus_dist, 'Sharp': sharp_dist}
    predicted_category = min(distances, key=distances.get)

    return predicted_category

# 분석 및 이미지 이름 변경 함수
def analyze_and_rename_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)

                # 각 측정값 계산
                blurriness = measure_blurriness(image_path)
                motion_blur_score, motion_direction = detect_motion_blur(image_path)
                (blur_values_8x8, blur_mean, blur_variance, high_blur_count,
                 variance_to_mean_ratio, max_min_diff, top_bottom_diff,
                 max_min_to_mean_ratio, max_min_ratio, top_bottom_ratio) = detect_depth_blur(image_path)

                # 이미지 분류
                classification = classify_image_weighted(
                    blurriness, motion_blur_score, blur_mean, blur_variance,
                    variance_to_mean_ratio, max_min_diff, max_min_ratio,
                    top_bottom_diff, max_min_to_mean_ratio, top_bottom_ratio
                )

                # 기존 접미사를 제거하고 새로운 접미사 추가
                base_name, ext = os.path.splitext(file)
                if base_name.endswith(('_0', '_1', '_2', '_unknown')):
                    base_name = base_name.rsplit('_', 1)[0]

                # 파일명 변경 (_0, _1, _2 추가)
                if classification == 'Blurry':
                    suffix = '_0'
                elif classification == 'Sharp':
                    suffix = '_1'
                elif classification == 'Shallow Focus':
                    suffix = '_2'
                else:
                    suffix = '_unknown'

                new_filename = f"{base_name}{suffix}{ext}"
                new_path = os.path.join(root, new_filename)

                try:
                    os.rename(image_path, new_path)
                    print(f"Renamed {image_path} to {new_filename}")
                except PermissionError:
                    print(f"Skipping {image_path}: Permission denied.")

# 사용 예시
analyze_and_rename_images("/Users/choizeus/Pictures/Photo")

