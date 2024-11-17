

# 이미지 분류기 기준을 책정하기 위한 이미지 분석기
# 설정된 폴더들을 순회하면서 해당 폴더 내의 이미지 분석 후 분석값 도출

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

    # 4x4 영역으로 분할하여 변동성 계산
    regions = [
        image[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
        for i in range(8) for j in range(8)
    ]
    # 소수점 한 자리로 제한하여 계산
    blur_values = [round(cv2.Laplacian(region, cv2.CV_64F).var(), 1) for region in regions]
    blur_values_8x8 = [blur_values[i * 8:(i + 1) * 8] for i in range(8)]

    # 전체 평균과 분산
    blur_mean = np.mean(blur_values)
    blur_variance = np.var(blur_values)

    # 최고값과 최저값의 차이
    max_min_diff = max(blur_values) - min(blur_values)
    max_min_to_mean_ratio = max_min_diff / blur_mean if blur_mean != 0 else 0

    # 상위 20%와 하위 20% 값의 차이
    top_20_percent = np.percentile(blur_values, 95)
    bottom_20_percent = np.percentile(blur_values, 20)
    top_bottom_diff = top_20_percent - bottom_20_percent

    # 최고값과 최저값의 차이 비율
    max_min_ratio = max(blur_values) / min(blur_values) if min(blur_values) != 0 else 0
    max_min_to_ratio_ratio = max_min_diff / blur_mean if blur_mean != 0 else 0

    # 상위 20%와 하위 20% 값의 차이 비율
    top_bottom_ratio = top_20_percent / bottom_20_percent if bottom_20_percent != 0 else 0



    # 평균 대비 분산 비율
    variance_to_mean_ratio = blur_variance / blur_mean if blur_mean != 0 else 0

    # 평균보다 높은 구역의 개수 계산
    high_blur_count = sum(1 for value in blur_values if value > blur_mean)

    # 상대적 기준: 평균 이상인 구역이 일정 개수 이상일 때 심도 흐림으로 판단

    depth_blur_detected = high_blur_count >= 10 and (variance_to_mean_ratio > 0.5)


    return blur_values_8x8, blur_mean, blur_variance, high_blur_count, variance_to_mean_ratio, max_min_diff, top_bottom_diff, max_min_to_mean_ratio, max_min_ratio, max_min_to_ratio_ratio, top_bottom_ratio


# 폴더 내 모든 파일에 대해 분석 수행
def analyze_images_in_folder(folder_path):
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file_name)
            blurriness = measure_blurriness(image_path)
            motion_blur_score, motion_direction = detect_motion_blur(image_path)
            depth_blur_values, blur_mean, blur_variance, high_blur_count, variance_to_mean_ratio, max_min_diff, top_bottom_diff, max_min_to_mean_ratio, max_min_ratio, max_min_to_ratio_ratio, top_bottom_ratio = detect_depth_blur(image_path)

            # 결과 리스트에 정보 추가
            results.append({
                'File Name': file_name,  # 파일 이름
                'Blurriness Score': blurriness,  # 흐림 점수
                'Motion Blur Score': motion_blur_score,  # 모션 블러 점수
                'Motion Blur Direction': motion_direction,  # 모션 블러 방향
                'Depth Blur Values': depth_blur_values,  # 심도 흐림 값
                'Depth Blur Values Mean': blur_mean,  # 64개의 평균 값
                'Depth Blur Values Variance': blur_variance,  # 64개의 분산 값
                'Count of Regions Above Mean': high_blur_count,  # 평균 이상 영역 개수
                'Variance to Mean Ratio': variance_to_mean_ratio,  # 평균 대비 분산 비율
                'Max-Min Difference': max_min_diff,
                'Max-Min Difference Ratio': max_min_to_mean_ratio,
                'Top-Bottom 20% Difference': top_bottom_diff,
                'Max-Min Ratio': max_min_ratio,
                'Max-Min Ratio Ratio': max_min_to_ratio_ratio,
                'Top-Bottom 20% Ratio': top_bottom_ratio

            })
    return results


# 이상치 제거 함수 (IQR 방식)
def remove_outliers(data):
    q1 = np.percentile(data, 10)
    q3 = np.percentile(data, 90)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]


# 폴더 내 결과들의 평균, 최댓값, 최솟값 계산 (이상치 제거 포함)
def calculate_folder_stats(results, include_outliers=True):
    # 각 항목별 데이터 리스트 수집
    blurriness_scores = [res['Blurriness Score'] for res in results]
    motion_blur_scores = [res['Motion Blur Score'] for res in results]
    depth_blur_means = [res['Depth Blur Values Mean'] for res in results]
    depth_blur_variances = [res['Depth Blur Values Variance'] for res in results]
    high_blur_counts = [res['Count of Regions Above Mean'] for res in results]
    variance_to_mean_ratios = [res['Variance to Mean Ratio'] for res in results]
    max_min_diffs = [res['Max-Min Difference'] for res in results]
    top_bottom_diffs = [res['Top-Bottom 20% Difference'] for res in results]
    max_min_ratios = [res['Max-Min Ratio'] for res in results]
    top_bottom_ratios = [res['Top-Bottom 20% Ratio'] for res in results]
    max_min_diff_ratios = [res['Max-Min Difference Ratio'] for res in results]
    max_min_ratio_ratios = [res['Max-Min Ratio Ratio'] for res in results]

    # 통계 계산을 위한 데이터를 준비 (이상치 제거 옵션 적용)
    def get_statistics(data):
        data = remove_outliers(data) if not include_outliers else data
        return np.mean(data), np.max(data), np.min(data)

    # 항목별 통계 계산
    stats = {
        'Blurriness Score': get_statistics(blurriness_scores),
        'Motion Blur Score': get_statistics(motion_blur_scores),
        'Depth Blur Values Mean': get_statistics(depth_blur_means),
        'Depth Blur Values Variance': get_statistics(depth_blur_variances),
        'Count of Regions Above Mean': get_statistics(high_blur_counts),
        'Variance to Mean Ratio': get_statistics(variance_to_mean_ratios),
        'Max-Min Difference': get_statistics(max_min_diffs),
        'Top-Bottom 20% Difference': get_statistics(top_bottom_diffs),
        'Max-Min Ratio': get_statistics(max_min_ratios),
        'Top-Bottom 20% Ratio': get_statistics(top_bottom_ratios),
        'Max-Min Difference Ratio': get_statistics(max_min_diff_ratios),
        'Max-Min Ratio Ratio': get_statistics(max_min_ratio_ratios)
    }

    return stats


# 세 폴더 각각 분석
folders = {
    'Blurry': './images/analytics/blurry',
    'Shallow Focus': './images/analytics/shallowFocus',
    'Sharp': './images/analytics/sharp'
}

for folder_name, folder_path in folders.items():
    print(f"\n--- {folder_name} 폴더의 이미지 분석 ---")
    results = analyze_images_in_folder(folder_path)
    for result in results:
        print(f"파일 이름: {result['File Name']}")
        print(f"  흐림 점수: {result['Blurriness Score']}")
        print(f"  모션 블러 점수: {result['Motion Blur Score']}")
        print(f"  모션 블러 방향: {result['Motion Blur Direction']}")
        print("  심도 흐림 값 (8x8):")
        for row in result['Depth Blur Values']:
            print("    ", row)
        print(f"  64개의 평균 값: {result['Depth Blur Values Mean']}")
        print(f"  64개의 분산 값: {result['Depth Blur Values Variance']}")
        print(f"  평균 이상 영역 개수: {result['Count of Regions Above Mean']}")
        print(f"  평균 대비 분산 비율: {result['Variance to Mean Ratio']}")
        print(f"  최고값-최저값 차이: {result['Max-Min Difference']}")
        print(f"  최고값-최저값 차이 비율: {result['Max-Min Difference Ratio']}")
        print(f"  상위 20% - 하위 20% 차이: {result['Top-Bottom 20% Difference']}")
        print(f"  최고값-최저값 비율: {result['Max-Min Ratio']}")
        print(f"  최고값-최저값 비율의 비율: {result['Max-Min Ratio Ratio']}")
        print(f"  상위 20% - 하위 20% 비율: {result['Top-Bottom 20% Ratio']}")
        print("-----------------------------------------------------")


# 폴더별 이미지 분석 결과 및 통계 출력 (이상치 제거 포함)
for folder_name, folder_path in folders.items():
    print(f"\n--- {folder_name} 폴더의 이미지 분석 ---")
    results = analyze_images_in_folder(folder_path)

    # 이상치 포함 통계 계산 및 출력
    stats_with_outliers = calculate_folder_stats(results, include_outliers=True)
    print(f"\n--- {folder_name} 폴더 내 모든 이미지의 통계 (이상치 포함) ---")
    for stat_name, (avg, max_val, min_val) in stats_with_outliers.items():
        print(f"  {stat_name}:")
        print(f"    평균: {avg}")
        print(f"    최댓값: {max_val}")
        print(f"    최솟값: {min_val}")

    # 이상치 제거 통계 계산 및 출력
    stats_without_outliers = calculate_folder_stats(results, include_outliers=False)
    print(f"\n--- {folder_name} 폴더 내 모든 이미지의 통계 (이상치 제거) ---")
    for stat_name, (avg, max_val, min_val) in stats_without_outliers.items():
        print(f"  {stat_name}:")
        print(f"    평균 (이상치 제거): {avg}")
        print(f"    최댓값 (이상치 제거): {max_val}")
        print(f"    최솟값 (이상치 제거): {min_val}")
    print("=====================================================")
