
# 이미지 분류기 성능 측정기
# 폴더 순회하면서 해당 폴더 내 정답률 측정


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


import itertools


# 폴더별 평가를 통한 최적의 조건 탐색 함수
def optimize_conditions(folders):
    best_accuracy = 0
    best_sharp_threshold = None
    best_shallow_threshold = None
    best_example = None

    # 0.7에서 1.5까지 0.1 단위로 테스트
    for example in np.arange(0.7, 1.6, 0.1):
        for sharp_threshold in range(10):  # 0에서 9까지 테스트
            for shallow_threshold in range(5):  # 0에서 4까지 테스트
                total_correct = 0
                total_incorrect = 0
                total_unknown = 0

                # 폴더별로 평가 수행
                for folder_name, folder_path in folders.items():
                    correct = 0
                    incorrect = 0
                    unknown = 0

                    for file_name in os.listdir(folder_path):
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(folder_path, file_name)

                            # 각 이미지에 대해 모든 분석 함수 호출
                            blurriness = measure_blurriness(image_path)
                            motion_blur_score, motion_direction = detect_motion_blur(image_path)
                            (blur_values_8x8, blur_mean, blur_variance, high_blur_count, variance_to_mean_ratio,
                             max_min_diff, top_bottom_diff, max_min_to_mean_ratio, max_min_ratio,
                             top_bottom_ratio) = detect_depth_blur(image_path)

                            # 조건에 따른 이미지 분류
                            predicted_category = classify_image_with_custom_thresholds(
                                blurriness, motion_blur_score, motion_direction, blur_values_8x8, blur_mean,
                                blur_variance,
                                high_blur_count, variance_to_mean_ratio, max_min_diff, top_bottom_diff,
                                max_min_to_mean_ratio,
                                max_min_ratio, top_bottom_ratio, sharp_threshold, shallow_threshold, example
                            )

                            # 분류 결과 확인
                            if predicted_category == folder_name:
                                correct += 1
                            elif predicted_category == 'Unknown':
                                unknown += 1
                            else:
                                incorrect += 1

                    total_correct += correct
                    total_incorrect += incorrect
                    total_unknown += unknown

                # 총 정답률 계산
                total_images = total_correct + total_incorrect + total_unknown
                accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0

                # 현재 조합에 대한 정답률 출력
                print(
                    f"Example: {example:.1f}, Sharp 조건: {sharp_threshold}, Shallow 조건: {shallow_threshold}, 정답률: {accuracy:.2f}%")

                # 정답률이 이전 최대 정답률보다 높다면 갱신
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_sharp_threshold = sharp_threshold
                    best_shallow_threshold = shallow_threshold
                    best_example = example

    # 최적 조건과 최대 정답률 출력
    print(f"\n최적 정답률: {best_accuracy:.2f}%")
    print(f"최적 sharp 조건: {best_sharp_threshold}")
    print(f"최적 shallow 조건: {best_shallow_threshold}")
    print(f"최적 example 값: {best_example:.1f}")


# 변경된 분류 함수 (조건 임계값을 매개변수로 받도록 수정)
def classify_image_with_custom_thresholds(blurriness, motion_blur_score, motion_direction, blur_values_8x8, blur_mean,
                                          blur_variance, high_blur_count, variance_to_mean_ratio, max_min_diff,
                                          top_bottom_diff, max_min_to_mean_ratio, max_min_ratio, top_bottom_ratio,
                                          sharp_threshold, shallow_threshold, example):
    # 조정된 수치 기준 설정
    adjusted_blurriness_threshold = 150 * example
    adjusted_motion_blur_score_blurry = 60000000 * example
    adjusted_blur_variance_blurry = 4500 * example
    adjusted_top_bottom_ratio_blurry = 30 * example

    adjusted_motion_blur_score_sharp = 200000000 * example
    adjusted_blur_mean_sharp = 900 * example
    adjusted_blur_variance_sharp = 4000000 * example
    adjusted_variance_to_mean_ratio_sharp = 3600 * example
    adjusted_max_min_diff_sharp = 5000 * example
    adjusted_top_bottom_diff_sharp = 2900 * example
    adjusted_max_min_ratio_sharp = 1700 * example
    adjusted_top_bottom_ratio_sharp_min = 150 * example
    adjusted_top_bottom_ratio_sharp_max = 750 * example

    adjusted_blurriness_shallow = 100 * example
    adjusted_blur_variance_shallow_min = 500 * example
    adjusted_blur_variance_shallow_max = 6000000 * example
    adjusted_max_min_to_mean_ratio_shallow = 17 * example
    adjusted_top_bottom_ratio_shallow = 8000 * example

    # 1. Blurry 기준 - 세 가지 조건 중 3개 이상 충족 시 True (기존 유지)
    blurry_conditions = [
        blurriness <= adjusted_blurriness_threshold,
        motion_blur_score > adjusted_motion_blur_score_blurry,
        blur_variance < adjusted_blur_variance_blurry,
        top_bottom_ratio < adjusted_top_bottom_ratio_blurry
    ]
    if sum(blurry_conditions) >= 3:
        return 'Blurry'

    # 2. Sharp 기준 - 조건 개수 sharp_threshold 이상 충족 시 True
    sharp_conditions = [
        blurriness > adjusted_blurriness_threshold,
        motion_blur_score < adjusted_motion_blur_score_sharp,
        blur_mean > adjusted_blur_mean_sharp,
        blur_variance > adjusted_blur_variance_sharp,
        variance_to_mean_ratio > adjusted_variance_to_mean_ratio_sharp,
        max_min_diff > adjusted_max_min_diff_sharp,
        top_bottom_diff > adjusted_top_bottom_diff_sharp,
        max_min_ratio > adjusted_max_min_ratio_sharp,
        adjusted_top_bottom_ratio_sharp_min < top_bottom_ratio < adjusted_top_bottom_ratio_sharp_max
    ]
    if sum(sharp_conditions) >= sharp_threshold:
        return 'Sharp'

    # 3. Shallow Focus 기준 - 조건 개수 shallow_threshold 이상 충족 시 True
    shallow_focus_conditions = [
        adjusted_blurriness_shallow < blurriness,
        adjusted_blur_variance_shallow_min < blur_variance < adjusted_blur_variance_shallow_max,
        max_min_to_mean_ratio > adjusted_max_min_to_mean_ratio_shallow,
        top_bottom_ratio > adjusted_top_bottom_ratio_shallow
    ]
    if sum(shallow_focus_conditions) >= shallow_threshold:
        return 'Shallow Focus'

    # 위 조건에 해당하지 않으면 Unknown으로 분류
    else:
        return 'Unknown'


# 폴더별 최적 조건 탐색
folders = {
    'Blurry': './images/analytics/blurry',
    'Shallow Focus': './images/analytics/shallowFocus',
    'Sharp': './images/analytics/sharp'
}

optimize_conditions(folders)
