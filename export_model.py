"""
학습된 모델(model.npz)을 웹 브라우저에서 사용할 수 있도록 JSON 파일로 변환하는 스크립트.
실행 결과로 docs/model.json 파일이 만들어진다.
(GitHub Pages는 저장소의 docs 폴더를 웹사이트로 배포할 수 있어서 docs 폴더에 저장한다.)
"""
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.npz")
OUTPUT_PATH = os.path.join(BASE_DIR, "docs", "model.json")

# 소수점 자릿수: 파일 크기를 줄이기 위해 반올림한다 (정확도에는 거의 영향 없음)
DECIMALS = 5


def export():
    """npz 파일의 가중치를 층별로 꺼내 JSON으로 저장한다."""
    data = np.load(MODEL_PATH)
    num_layers = len(data.files) // 2

    layers = []
    for i in range(num_layers):
        weights = data[f"W{i}"]  # 모양: (입력 개수, 출력 개수)
        bias = data[f"b{i}"]     # 모양: (출력 개수,)
        layers.append({
            "inputSize": int(weights.shape[0]),
            "outputSize": int(weights.shape[1]),
            # 2차원 배열을 1차원으로 펼쳐서 저장 (행 우선 순서: W[i][j] = weights[i * outputSize + j])
            "weights": np.round(weights.flatten(), DECIMALS).tolist(),
            "bias": np.round(bias, DECIMALS).tolist(),
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # 공백 없이 저장해서 파일 크기를 줄인다
        json.dump({"layers": layers}, f, separators=(",", ":"))

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"변환 완료: {OUTPUT_PATH} ({size_mb:.2f} MB)")
    for i, layer in enumerate(layers):
        print(f"  층 {i}: {layer['inputSize']} -> {layer['outputSize']}")


if __name__ == "__main__":
    export()
