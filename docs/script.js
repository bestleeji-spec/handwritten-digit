// 손글씨 숫자 인식 웹 버전
// 파이썬으로 학습한 모델(model.json)을 불러와 브라우저에서 직접 예측한다.

const CANVAS_SIZE = 280; // 그림판 크기 (픽셀)
const BRUSH_SIZE = 20;   // 붓 두께

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const clearButton = document.getElementById("clearButton");
const resultDigit = document.getElementById("resultDigit");
const resultText = document.getElementById("resultText");
const probList = document.getElementById("probList");

let model = null;       // 불러온 모델 가중치
let isDrawing = false;  // 현재 그리는 중인지 여부
let lastPoint = null;   // 직전 마우스 위치

// ---------------------------------------------------------------
// 1. 모델 불러오기
// ---------------------------------------------------------------

async function loadModel() {
  try {
    const response = await fetch("model.json");
    const data = await response.json();
    // 계산 속도를 위해 일반 배열을 Float32Array로 바꾼다
    model = data.layers.map((layer) => ({
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      weights: Float32Array.from(layer.weights),
      bias: Float32Array.from(layer.bias),
    }));
    resultText.textContent = "숫자를 그려보세요";
  } catch (error) {
    // 파일을 더블클릭해서 열면 보안 정책 때문에 model.json을 읽지 못한다
    resultText.textContent = "모델을 불러오지 못했습니다. 로컬 서버로 실행해 주세요.";
    console.error(error);
  }
}

// ---------------------------------------------------------------
// 2. 신경망 예측 (파이썬 train.py의 forward 함수와 같은 계산)
// ---------------------------------------------------------------

function predict(input) {
  let activation = input;
  model.forEach((layer, layerIndex) => {
    const { inputSize, outputSize, weights, bias } = layer;
    const output = new Float32Array(outputSize);
    // 행렬 곱셈: output[j] = bias[j] + sum(activation[i] * W[i][j])
    for (let j = 0; j < outputSize; j++) {
      output[j] = bias[j];
    }
    for (let i = 0; i < inputSize; i++) {
      const value = activation[i];
      if (value === 0) continue; // 0이면 곱해도 0이므로 건너뛴다 (속도 향상)
      const rowStart = i * outputSize;
      for (let j = 0; j < outputSize; j++) {
        output[j] += value * weights[rowStart + j];
      }
    }
    const isLastLayer = layerIndex === model.length - 1;
    if (!isLastLayer) {
      // 은닉층: ReLU (음수는 0으로)
      for (let j = 0; j < outputSize; j++) {
        output[j] = Math.max(0, output[j]);
      }
    }
    activation = output;
  });
  return softmax(activation);
}

function softmax(values) {
  // 수치 안정성을 위해 최댓값을 뺀 뒤 지수 함수 계산
  const max = Math.max(...values);
  const exps = Array.from(values, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// ---------------------------------------------------------------
// 3. 전처리: 그림을 MNIST 형식(28x28, 가운데 정렬)으로 변환
//    (파이썬 app.py의 preprocess 함수와 같은 순서)
// ---------------------------------------------------------------

function preprocess() {
  // 그림판의 픽셀을 0~1 밝기 값 배열로 읽는다 (빨강 채널 사용, 흰 글씨 = 1)
  const pixels = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data;
  const gray = new Float32Array(CANVAS_SIZE * CANVAS_SIZE);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = pixels[i * 4] / 255;
  }

  // 1) 숫자가 있는 영역(경계 상자)을 찾는다
  let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = -1, maxY = -1;
  for (let y = 0; y < CANVAS_SIZE; y++) {
    for (let x = 0; x < CANVAS_SIZE; x++) {
      if (gray[y * CANVAS_SIZE + x] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null; // 아무것도 그리지 않음

  // 2) 비율을 유지한 채 긴 쪽이 20픽셀이 되도록 축소한다 (영역 평균 방식)
  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const scale = 20 / Math.max(boxW, boxH);
  const newW = Math.max(1, Math.round(boxW * scale));
  const newH = Math.max(1, Math.round(boxH * scale));
  const small = resizeArea(gray, minX, minY, boxW, boxH, newW, newH);

  // 3) 28x28 검은 배경의 가운데에 붙인다
  let digit = new Float32Array(28 * 28);
  const offsetX = Math.floor((28 - newW) / 2);
  const offsetY = Math.floor((28 - newH) / 2);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      digit[(y + offsetY) * 28 + (x + offsetX)] = small[y * newW + x];
    }
  }
  digit = gaussianBlur(digit);

  // 4) 무게중심이 정중앙(14, 14)에 오도록 이동한다
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = digit[y * 28 + x];
      total += v;
      sumX += x * v;
      sumY += y * v;
    }
  }
  const shiftX = Math.round(14 - sumX / total);
  const shiftY = Math.round(14 - sumY / total);
  return shiftImage(digit, shiftX, shiftY);
}

// 원본 영역을 작은 크기로 줄인다: 작은 이미지의 한 칸 = 원본에서 그 칸이 덮는 영역의 평균
function resizeArea(src, startX, startY, srcW, srcH, dstW, dstH) {
  const dst = new Float32Array(dstW * dstH);
  for (let dy = 0; dy < dstH; dy++) {
    const y0 = startY + Math.floor((dy * srcH) / dstH);
    const y1 = startY + Math.max(Math.floor(((dy + 1) * srcH) / dstH), y0 - startY + 1);
    for (let dx = 0; dx < dstW; dx++) {
      const x0 = startX + Math.floor((dx * srcW) / dstW);
      const x1 = startX + Math.max(Math.floor(((dx + 1) * srcW) / dstW), x0 - startX + 1);
      let sum = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          sum += src[y * CANVAS_SIZE + x];
        }
      }
      dst[dy * dstW + dx] = sum / ((y1 - y0) * (x1 - x0));
    }
  }
  return dst;
}

// 가장자리를 살짝 부드럽게 만든다 (3x3 가우시안 블러, 시그마 0.5)
function gaussianBlur(img) {
  const kernel = [0.1065, 0.787, 0.1065];
  const temp = new Float32Array(28 * 28);
  const out = new Float32Array(28 * 28);
  // 가로 방향으로 한 번, 세로 방향으로 한 번 적용
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let sum = 0;
      for (let k = -1; k <= 1; k++) {
        const xx = x + k;
        if (xx >= 0 && xx < 28) sum += img[y * 28 + xx] * kernel[k + 1];
      }
      temp[y * 28 + x] = sum;
    }
  }
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let sum = 0;
      for (let k = -1; k <= 1; k++) {
        const yy = y + k;
        if (yy >= 0 && yy < 28) sum += temp[yy * 28 + x] * kernel[k + 1];
      }
      out[y * 28 + x] = sum;
    }
  }
  return out;
}

// 이미지를 (dx, dy)만큼 이동한다. 밖으로 나간 부분은 버리고 빈 곳은 0으로 채운다
function shiftImage(img, dx, dy) {
  const out = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
        out[ny * 28 + nx] = img[y * 28 + x];
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------
// 4. 그림 그리기 (마우스와 터치 모두 지원)
// ---------------------------------------------------------------

// 화면상의 위치를 캔버스 내부 좌표로 바꾼다 (모바일에서 캔버스가 작아져도 정확하게)
function getPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * CANVAS_SIZE) / rect.width,
    y: ((event.clientY - rect.top) * CANVAS_SIZE) / rect.height,
  };
}

function drawLine(from, to) {
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = BRUSH_SIZE;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

canvas.addEventListener("pointerdown", (event) => {
  isDrawing = true;
  canvas.setPointerCapture(event.pointerId);
  lastPoint = getPoint(event);
  drawLine(lastPoint, lastPoint); // 클릭만 해도 점이 찍히도록
});

canvas.addEventListener("pointermove", (event) => {
  if (!isDrawing) return;
  const point = getPoint(event);
  drawLine(lastPoint, point);
  lastPoint = point;
});

canvas.addEventListener("pointerup", () => {
  if (!isDrawing) return;
  isDrawing = false;
  recognize(); // 손을 떼면 바로 인식
});

clearButton.addEventListener("click", clearCanvas);

// ---------------------------------------------------------------
// 5. 인식 실행 및 결과 표시
// ---------------------------------------------------------------

function recognize() {
  if (!model) return;
  const input = preprocess();
  if (!input) return;
  const probs = predict(input);
  const best = probs.indexOf(Math.max(...probs));
  resultDigit.textContent = best;
  resultText.textContent = `확신도: ${(probs[best] * 100).toFixed(1)}%`;
  renderProbs(probs);
}

// 0~9 각 숫자의 확률을 막대그래프로 보여준다
function renderProbs(probs) {
  const best = probs.some((p) => p > 0) ? probs.indexOf(Math.max(...probs)) : -1;
  probList.innerHTML = "";
  probs.forEach((p, digit) => {
    const item = document.createElement("li");
    if (digit === best) item.className = "best";
    item.innerHTML = `
      <span>${digit}</span>
      <div class="bar-track"><div class="bar-fill" style="width: ${p * 100}%"></div></div>
      <span class="prob-value">${Math.round(p * 100)}%</span>`;
    probList.appendChild(item);
  });
}

function clearCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  resultDigit.textContent = "?";
  resultText.textContent = model ? "숫자를 그려보세요" : resultText.textContent;
  renderProbs(new Array(10).fill(0));
}

// 시작할 때 그림판을 초기화하고 모델을 불러온다
clearCanvas();
loadModel();
