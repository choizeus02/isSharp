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
    return total_edges


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

# 가중치 기반 분류 함수 (유클리드 거리 계산에 추가된 파라미터 포함)
def classify_image_weighted(blurriness, motion_blur_score, blur_mean, blur_variance,
                            variance_to_mean_ratio, max_min_diff, max_min_ratio,
                            top_bottom_diff, max_min_to_mean_ratio, top_bottom_ratio):
    weights = {
        'blurriness': 30,
        'motion_blur_score': 1,
        'blur_mean': 2,
        'blur_variance': 100,
        'variance_to_mean_ratio': 15,
        'max_min_diff': 5,
        'max_min_ratio': 3,
        'top_bottom_diff': 30,
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

# 폴더 순회 및 평가 함수
def evaluate_folders_weighted(folders):
    total_correct = 0
    total_incorrect = 0
    total_unknown = 0

    for folder_name, folder_path in folders.items():
        correct = 0
        incorrect = 0
        unknown = 0
        blurry_count = 0
        shallow_focus_count = 0
        sharp_count = 0

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, file_name)

                # 각 이미지의 지표 계산
                blurriness = measure_blurriness(image_path)
                motion_blur_score = detect_motion_blur(image_path)
                (blur_values_8x8, blur_mean, blur_variance, high_blur_count, variance_to_mean_ratio,
                 max_min_diff, top_bottom_diff, max_min_to_mean_ratio, max_min_ratio,
                 top_bottom_ratio) = detect_depth_blur(image_path)

                # 가중치 기반 분류 함수 호출
                predicted_category = classify_image_weighted(
                    blurriness, motion_blur_score, blur_mean, blur_variance,
                    variance_to_mean_ratio, max_min_diff, max_min_ratio,
                    top_bottom_diff, max_min_to_mean_ratio, top_bottom_ratio
                )

                # 결과 평가
                if predicted_category == folder_name:
                    correct += 1
                elif predicted_category == 'Unknown':
                    unknown += 1
                else:
                    incorrect += 1

                # 각 분류별 카운트
                if predicted_category == 'Blurry':
                    blurry_count += 1
                elif predicted_category == 'Shallow Focus':
                    shallow_focus_count += 1
                elif predicted_category == 'Sharp':
                    sharp_count += 1

        total_correct += correct
        total_incorrect += incorrect
        total_unknown += unknown

        total = correct + incorrect + unknown
        accuracy = (correct / total) * 100 if total > 0 else 0
        unknown_ratio = (unknown / total) * 100 if total > 0 else 0
        print(f"\n--- {folder_name} 폴더 ---")
        print(f"정답 수: {correct}")
        print(f"오답 수: {incorrect}")
        print(f"Blurry 수: {blurry_count}")
        print(f"Shallow Focus 수: {shallow_focus_count}")
        print(f"Sharp 수: {sharp_count}")
        print(f"Unknown 수: {unknown}")
        print(f"정답률: {accuracy:.2f}%")
        print(f"Unknown 비율: {unknown_ratio:.2f}%")

    total_images = total_correct + total_incorrect + total_unknown
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    overall_unknown_ratio = (total_unknown / total_images) * 100 if total_images > 0 else 0
    print(f"\n전체 정답 수: {total_correct}")
    print(f"전체 오답 수: {total_incorrect}")
    print(f"전체 Unknown 수: {total_unknown}")
    print(f"전체 정답률: {overall_accuracy:.2f}%")
    print(f"전체 Unknown 비율: {overall_unknown_ratio:.2f}%")


# 폴더별 평가 수행
folders = {
    'Blurry': './images/analytics/blurry',
    'Shallow Focus': './images/analytics/shallowFocus',
    'Sharp': './images/analytics/sharp'
}

evaluate_folders_weighted(folders)
