"""
손글씨 숫자 인식 GUI 프로그램.
마우스로 캔버스에 숫자를 그리면 학습된 모델이 어떤 숫자인지 예측한다.
model.npz가 없으면 먼저 train.py의 학습을 자동으로 실행한다.
"""
import os
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from train import MODEL_PATH, predict, train

CANVAS_SIZE = 280   # 그림판 크기 (픽셀)
BRUSH_SIZE = 20     # 붓 두께


def load_model():
    """저장된 모델을 불러오고, 없으면 새로 학습한다."""
    if not os.path.exists(MODEL_PATH):
        print("저장된 모델이 없어 학습을 시작합니다...")
        return train()
    data = np.load(MODEL_PATH)
    return {k: data[k] for k in data.files}


def preprocess(image):
    """
    그림판 이미지를 MNIST와 같은 형식(28x28, 중앙 정렬)으로 변환한다.
    MNIST는 숫자를 20x20 상자에 맞춘 뒤 무게중심을 28x28의 가운데로 옮긴 형태다.
    """
    arr = np.array(image)
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return None

    # 1) 숫자가 있는 영역만 잘라낸다
    cropped = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    # 2) 비율을 유지한 채 긴 쪽이 20픽셀이 되도록 축소한다
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    # 3) 28x28 검은 배경의 가운데에 붙인다
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.5))

    # 4) 무게중심이 정중앙(14, 14)에 오도록 이동한다
    digit = np.array(canvas, dtype=np.float32) / 255.0
    total = digit.sum()
    rows, cols = np.indices(digit.shape)
    cy, cx = (rows * digit).sum() / total, (cols * digit).sum() / total
    dy, dx = int(round(14 - cy)), int(round(14 - cx))
    digit = np.roll(np.roll(digit, dy, axis=0), dx, axis=1)

    return digit.reshape(1, 784)


class DigitApp:
    """그림판과 예측 결과를 보여주는 메인 창."""

    def __init__(self, root, params):
        self.root = root
        self.params = params
        self.last_point = None
        root.title("손글씨 숫자 인식")
        root.resizable(False, False)

        # 화면에 보이는 캔버스 (검은 배경에 흰 글씨)
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, rowspan=3, padx=10, pady=10)

        # 예측에 사용할 실제 이미지 (화면 캔버스와 동일하게 그린다)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        # 마우스 이벤트 연결
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # 예측 결과 표시 영역
        self.result_label = tk.Label(root, text="?", font=("Arial", 72, "bold"), width=3)
        self.result_label.grid(row=0, column=1, padx=10)
        self.conf_label = tk.Label(root, text="숫자를 그려보세요", font=("Malgun Gothic", 11))
        self.conf_label.grid(row=1, column=1)

        # 0~9 각 숫자의 확률 막대그래프
        self.bars = tk.Canvas(root, width=220, height=200)
        self.bars.grid(row=2, column=1, padx=10)

        tk.Button(root, text="지우기", font=("Malgun Gothic", 11), command=self.clear).grid(
            row=3, column=0, columnspan=2, pady=(0, 10), ipadx=20)

        self.draw_bars(np.zeros(10))

    def on_press(self, event):
        """마우스를 누르면 점을 찍는다."""
        self.last_point = (event.x, event.y)
        self.paint_line(event.x, event.y, event.x, event.y)

    def on_drag(self, event):
        """마우스를 끌면 이전 위치와 현재 위치를 선으로 잇는다."""
        x0, y0 = self.last_point
        self.paint_line(x0, y0, event.x, event.y)
        self.last_point = (event.x, event.y)

    def on_release(self, _event):
        """마우스를 떼면 예측을 실행한다."""
        self.last_point = None
        self.recognize()

    def paint_line(self, x0, y0, x1, y1):
        """화면 캔버스와 내부 이미지 양쪽에 둥근 선을 그린다."""
        r = BRUSH_SIZE / 2
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=BRUSH_SIZE, capstyle=tk.ROUND, smooth=True)
        self.draw.line((x0, y0, x1, y1), fill=255, width=BRUSH_SIZE)
        self.draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=255)

    def recognize(self):
        """현재 그림을 전처리하고 모델로 숫자를 예측한다."""
        x = preprocess(self.image)
        if x is None:
            return
        probs = predict(self.params, x)[0]
        digit = int(probs.argmax())
        self.result_label.config(text=str(digit))
        self.conf_label.config(text=f"확신도: {probs[digit] * 100:.1f}%")
        self.draw_bars(probs)

    def draw_bars(self, probs):
        """0~9 숫자별 확률을 가로 막대로 그린다."""
        self.bars.delete("all")
        best = int(probs.argmax()) if probs.any() else -1
        for i, p in enumerate(probs):
            y = i * 20 + 2
            self.bars.create_text(10, y + 8, text=str(i), font=("Arial", 10))
            self.bars.create_rectangle(25, y + 2, 25 + 150, y + 15, outline="#ccc")
            color = "#2a7de1" if i == best else "#9bbce6"
            self.bars.create_rectangle(25, y + 2, 25 + 150 * p, y + 15, fill=color, outline="")
            self.bars.create_text(215, y + 8, text=f"{p * 100:.0f}%", anchor="e", font=("Arial", 9))

    def clear(self):
        """그림판과 결과를 초기화한다."""
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)
        self.result_label.config(text="?")
        self.conf_label.config(text="숫자를 그려보세요")
        self.draw_bars(np.zeros(10))


if __name__ == "__main__":
    model = load_model()
    window = tk.Tk()
    DigitApp(window, model)
    window.mainloop()
