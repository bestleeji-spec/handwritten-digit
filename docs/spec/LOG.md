# LOG — 논의 로그와 기술 로그

작성일: 2026-11-04

## 논의 로그

2026년 11월 4일, 사용자와 Claude가 나눈 논의를 요청 순서대로 기록했다.

| # | 사용자 요청 | 논의·결정 | 결과 |
| --- | --- | --- | --- |
| 1 | 손글씨 숫자 인식 코드를 만들어 실행. 코드는 영어, 주석은 한글 | 딥러닝 라이브러리가 없어 NumPy로 MLP 직접 구현, Tkinter 그림판 | `train.py`, `app.py`, 정확도 98.73%, 창 실행 |
| 2 | requirements.txt 만들기 | Tkinter는 기본 포함이라 제외 | `numpy>=2.0`, `Pillow>=10.0` |
| 3 | 웹 버전 제작 방법 논의 (코드 작성 없이 대안 4개) | ① 순수 JS ② Flask/FastAPI ③ Gradio ④ CNN+ONNX 제시. 학습 목적이면 ①·② 추천 | 비교표 제공 |
| 4 | GitHub Pages 배포 계획으로 ① 선택. JSON 변환 → 웹 파일 → 로컬 테스트 순서로 진행 | Pages가 `docs` 폴더를 배포할 수 있어 웹 파일을 `docs`에 둠 | `export_model.py`, `docs/` 4개 파일, 로컬 서버 실행 |
| 5 | 테스트 결과 잘 됨. GitHub Pages 배포 방법을 1단계부터 안내 | Git·gh 설치와 로그인 상태 확인 후 6단계 안내, 직접 할지 맡길지 선택 요청 | 배포 가이드 |
| 6 | 2번(자동 진행) 선택 | 공개 저장소 생성이 권한 정책에 막힘. 1·2단계만 자동 진행하고 3·4단계는 사용자가 `!` 명령으로 직접 실행 | 저장소 생성, Pages 활성화 |
| 7 | 배포 사이트 정상 동작 확인, 감사 인사 | README, CNN, 모델 경량화를 다음 단계로 제안 | — |
| 8 | 지금까지 내용을 문서로 정리 | 도표(에폭별 정확도)와 다이어그램(파일 흐름) 포함 | [손글씨 숫자 인식 프로젝트 정리](https://claude.ai/artifact/9pdjrMhSEoPusgutmd4HH7) |
| 9 | PRD, TRD, TASK, 논의 로그, 기술 로그 작성 | 한 문서에 탭 5개로 구성 | [PRD·TRD·TASK·로그 문서](https://claude.ai/artifact/RUGFL3VnDabMdng2UCTsGz) |
| 10 | 문서 내용을 PRD.md, TRD.md, TASK.md, LOG.md 파일로 저장 | 논의 로그와 기술 로그는 LOG.md 하나로 합침, TRD 다이어그램은 Mermaid로 변환 | 이 파일들 |

## 기술 로그

### 환경과 실행 기록

**개발 환경**: Windows 11 Home, Python 3.13, NumPy 2.5.4, Pillow 13.0.0, Tkinter, Git 2.56.0, GitHub CLI 2.102.1 (계정 `bestleeji-spec`). PyTorch, TensorFlow, scikit-learn, Node.js는 없음.

| # | 작업 | 명령·방법 | 결과 |
| --- | --- | --- | --- |
| 1 | 패키지 확인 | `importlib.util.find_spec` | numpy·PIL·tkinter만 있음 |
| 2 | 모델 학습 | `python train.py` | MNIST 4개 파일 다운로드, 12에폭, 테스트 96.27% → 98.73%, 10분 넘게 걸려 백그라운드로 전환 |
| 3 | 전처리 검증 | 코드로 그린 1·7·0 예측 | 3개 모두 정답 |
| 4 | 데스크톱 실행 | `python app.py` | 창 실행 |
| 5 | JSON 변환 | `python export_model.py` | `docs/model.json` 4.66 MB, 층 784→256→128→10 |
| 6 | JSON 정확도 검증 | JSON 가중치로 테스트셋 예측 | 98.73%, 원본과 동일 |
| 7 | 로컬 웹 서버 | `python -m http.server 8000 --directory docs` | `/`, `/model.json` 모두 200 |
| 8 | Git 초기화·커밋 | `git init -b main`, `git add .`, `git commit` | 11개 파일, 커밋 `51117b8` |
| 9 | 저장소 생성 (사용자 실행) | `gh repo create handwritten-digit --public --source=. --push` | `main` 브랜치 푸시 |
| 10 | Pages 활성화 (사용자 실행) | `gh api repos/bestleeji-spec/handwritten-digit/pages -X POST ...` | `main` / `/docs`, HTTPS 강제 |
| 11 | 배포 확인 | Pages 빌드 상태 조회 + `curl` | `built`, 페이지·`model.json`·`script.js` 모두 200 |

### 문제와 해결

| 문제 | 원인 | 해결 |
| --- | --- | --- |
| 딥러닝 라이브러리 없음 | PyTorch·TensorFlow 미설치 | NumPy로 순전파·역전파·Adam을 직접 구현 |
| 학습 중 로그가 안 보임 | 출력 버퍼링 | 학습 완료 후 로그 파일로 확인 |
| 직접 그린 숫자가 MNIST와 모양이 다름 | 크기·위치가 제각각 | 자르기·20px 축소·가운데 배치·무게중심 정렬 전처리. 학습 때 최대 2px 이동 증강 |
| Node.js 없어 JS를 여기서 실행할 수 없음 | 미설치 | JSON 정확도는 파이썬으로 검증, 웹 동작은 사용자가 브라우저에서 확인 |
| `index.html` 더블클릭 시 모델 로딩 실패 | `file://`에서는 `fetch` 차단 | 로컬 HTTP 서버로 실행, 화면에 안내 문구 표시 |
| 공개 저장소 생성 명령 차단 | Claude Code 자동 모드의 외부 공개 제한 | 사용자가 `!` 명령으로 직접 실행 |
| `git add` 시 LF→CRLF 경고 | Windows 줄바꿈 설정 | 동작에 영향 없어 그대로 둠 |
| Git `user.name`이 이메일 주소 | 전역 설정 값 | 변경하지 않음. TASK의 선택 작업으로 남김 |
