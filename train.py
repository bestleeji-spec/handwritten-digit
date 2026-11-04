"""
MNIST 데이터셋으로 손글씨 숫자 인식 신경망(MLP)을 학습하는 스크립트.
외부 딥러닝 프레임워크 없이 NumPy만 사용한다.
학습이 끝나면 가중치를 model.npz 파일로 저장한다.
"""
import gzip
import os
import urllib.request

import numpy as np

# 데이터와 모델을 저장할 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model.npz")

# MNIST 다운로드 주소 (Google Cloud 미러)
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    """MNIST 파일이 없으면 내려받는다."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in MNIST_FILES.values():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"다운로드 중: {name}")
            urllib.request.urlretrieve(MNIST_URL + name, path)


def load_images(filename):
    """idx 형식의 이미지 파일을 읽어 (N, 784) 실수 배열로 반환한다."""
    with gzip.open(os.path.join(DATA_DIR, filename), "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 0~255 값을 0~1 범위로 정규화
    return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0


def load_labels(filename):
    """idx 형식의 라벨 파일을 읽어 정수 배열로 반환한다."""
    with gzip.open(os.path.join(DATA_DIR, filename), "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)


def init_params(layer_sizes, rng):
    """He 초기화로 각 층의 가중치와 편향을 만든다."""
    params = {}
    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        params[f"W{i}"] = (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        params[f"b{i}"] = np.zeros(fan_out, dtype=np.float32)
    return params


def forward(params, x):
    """순전파: 은닉층은 ReLU, 출력층은 소프트맥스. 역전파용 중간값도 함께 반환한다."""
    num_layers = len(params) // 2
    activations = [x]
    for i in range(num_layers):
        z = activations[-1] @ params[f"W{i}"] + params[f"b{i}"]
        if i < num_layers - 1:
            activations.append(np.maximum(z, 0))
        else:
            # 수치 안정성을 위해 최댓값을 빼고 소프트맥스 계산
            z = z - z.max(axis=1, keepdims=True)
            exp = np.exp(z)
            activations.append(exp / exp.sum(axis=1, keepdims=True))
    return activations


def predict(params, x):
    """입력 이미지들에 대한 클래스별 확률을 반환한다."""
    return forward(params, x)[-1]


def backward(params, activations, y):
    """역전파: 교차 엔트로피 손실에 대한 각 파라미터의 기울기를 계산한다."""
    num_layers = len(params) // 2
    batch_size = y.shape[0]
    grads = {}
    # 소프트맥스 + 교차 엔트로피의 기울기는 (예측 - 정답 원핫)
    delta = activations[-1].copy()
    delta[np.arange(batch_size), y] -= 1
    delta /= batch_size
    for i in reversed(range(num_layers)):
        grads[f"W{i}"] = activations[i].T @ delta
        grads[f"b{i}"] = delta.sum(axis=0)
        if i > 0:
            # ReLU 미분: 활성값이 0보다 큰 곳만 기울기 전달
            delta = (delta @ params[f"W{i}"].T) * (activations[i] > 0)
    return grads


def augment(batch, rng):
    """데이터 증강: 이미지를 상하좌우로 최대 2픽셀 무작위 이동시킨다."""
    images = batch.reshape(-1, 28, 28)
    out = np.empty_like(images)
    shifts = rng.integers(-2, 3, size=(len(images), 2))
    for k, (dy, dx) in enumerate(shifts):
        out[k] = np.roll(np.roll(images[k], dy, axis=0), dx, axis=1)
    return out.reshape(-1, 784)


def train(epochs=12, batch_size=128, lr=1e-3, seed=0):
    """Adam 옵티마이저로 MLP를 학습하고 모델을 저장한다."""
    download_mnist()
    x_train = load_images(MNIST_FILES["train_images"])
    y_train = load_labels(MNIST_FILES["train_labels"])
    x_test = load_images(MNIST_FILES["test_images"])
    y_test = load_labels(MNIST_FILES["test_labels"])
    print(f"학습 데이터: {x_train.shape}, 테스트 데이터: {x_test.shape}")

    rng = np.random.default_rng(seed)
    params = init_params([784, 256, 128, 10], rng)

    # Adam 옵티마이저 상태값 (1차, 2차 모멘트)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(val) for k, val in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    for epoch in range(1, epochs + 1):
        # 매 에폭마다 학습 데이터 순서를 섞는다
        order = rng.permutation(len(x_train))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = augment(x_train[idx], rng)
            yb = y_train[idx]

            activations = forward(params, xb)
            probs = activations[-1]
            total_loss += -np.log(probs[np.arange(len(yb)), yb] + 1e-12).sum()

            grads = backward(params, activations, yb)
            step += 1
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1 ** step)
                v_hat = v[k] / (1 - beta2 ** step)
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # 테스트 데이터로 정확도 측정
        test_acc = (predict(params, x_test).argmax(axis=1) == y_test).mean()
        print(f"에폭 {epoch:2d}/{epochs} | 평균 손실 {total_loss / len(x_train):.4f} | 테스트 정확도 {test_acc * 100:.2f}%")

        # 후반부에는 학습률을 낮춰 미세 조정
        if epoch == epochs // 2:
            lr *= 0.3

    np.savez(MODEL_PATH, **params)
    print(f"모델 저장 완료: {MODEL_PATH}")
    return params


if __name__ == "__main__":
    train()
