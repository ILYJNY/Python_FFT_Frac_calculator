# Python
---
# Codes with  annonation
import cv2  # OpenCV: 이미지 읽기/처리 (디코딩, 컨투어, 색공간 변환 등)
import numpy as np  # NumPy: 수치 연산 및 배열 처리
import requests  # requests: HTTP로 이미지/데이터를 다운로드할 때 사용
import matplotlib.pyplot as plt  # Matplotlib: 마스크/경로/결과 시각화용
import math  # math: 표준 수학 함수(제곱근, 로그, 삼각함수 등)
import cmath  # cmath: 복소수 연산용 함수 (필요 시 사용 가능)
import re  # re: 정규표현식 (문자열 전처리)
import sys  # sys: 시스템 관련 유틸 (필요 시 사용)

# ---------------- 기능: 소인수분해 ----------------
# fractorization(): 사용자로부터 정수를 입력받아 소인수분해 결과를 출력
def fractorization():
    # pen(n): 현재 n에서 가장 작은 소인수(p)와 그 지수(e), 그리고 남은 몫(n_remain)을 반환
    def pen(n):
        # 2부터 sqrt(n)까지 가능한 약수 i를 순회
        for i in range(2, math.floor(math.sqrt(n)) + 1):
            # i가 n을 나누면 소인수 후보
            if n % i == 0:
                # i의 거듭제곱이 몇 번 들어가는지 확인하기 위해 j를 증가시킴
                for j in range(0, math.floor(math.log(n, i)) + 2):
                    # 더 이상 i**j로 나누어떨어지지 않으면 j-1이 최대 지수
                    if n % i ** j != 0:
                        # 반환: (소인수 i, 지수 j-1, 남은 몫)
                        return i, j-1, n / (i ** (j-1))
        # 반복문에서 찾지 못하면 n은 소수임
        return n, 1, 1

    put = int(input('소인수분해할 정수 입력: '))  # 사용자 입력을 정수로 변환
    n = put  # 현재 분해 대상 변수 n (반복하면서 줄어듦)
    primelist = []  # 발견된 소인수 저장 리스트
    powerlist = []  # 각 소인수의 지수 저장 리스트
    while True:
        # pen을 호출하여 p(소인수), e(지수), n(나머지)를 얻음
        p, e, n = pen(n)
        primelist.append(p)  # 소인수 추가
        powerlist.append(e)  # 지수 추가
        # n이 1이면 모든 소인수 추출 완료
        if n == 1:
            break

    print("\n===== 소인수분해 결과 =====")
    # 결과 포맷 출력: 소수인지 또는 소인수 곱 형태인지
    if len(primelist) == 1:
        # 리스트에 하나만 있으면 입력값 자체가 소수
        print(f'{primelist[0]}^1 (소수)')
    else:
        # 여러 소인수인 경우 x 기호로 연결하여 출력
        for i in range(len(primelist)-1):
            print(f'{primelist[i]}^{powerlist[i]} x', end=' ')
        # 마지막 항 출력
        print(f'{int(primelist[-1])}^{powerlist[-1]} (소수가 아님)')

# ---------------- 기능: 계산기 (Colab/PC) ----------------
# safe_dict: eval 실행 시 허용할 안전한 이름(상수·함수)만 등록
safe_dict = {
    'pi': math.pi, 'π': math.pi, 'e': math.e, 'i': 1j, 'j': 1j,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'ln': math.log, 'log': lambda x, base=10: math.log(x, base),
    'sqrt': math.sqrt, 'abs': abs, 'mod': lambda x, y: x%y,
    'gamma': math.gamma, 'exp': math.exp, 'pow': pow
}

# preprocess: 사용자 수식을 eval 친화적(및 안전)하게 전처리
# - '^' → '**', '%' → ' mod ', 'π' → 'pi'
# - 'n!' → 'gamma(n+1)', '|...|' → 'abs(...)'
def preprocess(expr):
    expr = expr.replace('^','**').replace('%',' mod ').replace('π','pi')
    # 정규표현식으로 정수 팩토리얼 표기 변환
    expr = re.sub(r'(\\d+)!', r'gamma(\\1+1)', expr)
    # 절댓값 표기 |x|을 abs(x)로 변환
    expr = re.sub(r'\|([^\|]+)\|', r'abs(\\1)', expr)
    return expr

# calc: 전처리 후 안전한 환경에서 문자열 수식을 평가
def calc(expr):
    expr = preprocess(expr)  # 전처리
    try:
        # __builtins__를 None으로 해서 위험한 내장함수 접근 차단
        return eval(expr, {"__builtins__":None}, safe_dict)
    except Exception as e:
        # 오류는 문자열로 반환하여 GUI/터미널에서 보기 쉽게 함
        return f"계산 오류: {e}"

# init_calculator_colab: 구글 콜랩 환경이면 위젯 기반 계산기 표시
def init_calculator_colab():
    try:
        import google.colab  # 콜랩 전용 모듈이 있으면 콜랩 환경
        IN_COLAB = True
    except:
        IN_COLAB = False

    if not IN_COLAB:
        return False  # 콜랩이 아니면 False 반환

    from ipywidgets import Textarea, Button, Output, VBox  # 콜랩 위젯
    from IPython.display import display

    # 입력 위젯(여러 줄), 출력 위젯, 계산 버튼 생성
    input_area = Textarea(placeholder="계산식을 입력하세요...", layout={'width':'50%', 'height':'150px'})
    output_area = Output(layout={'border':'1px solid black', 'width':'50%', 'height':'150px'})
    calc_btn = Button(description="계산")
    history = []  # 계산 이력 저장소

    # 버튼 클릭 콜백: 입력 텍스트의 각 줄을 계산하여 출력 위젯에 표시
    def on_calc_click(b):
        output_area.clear_output()  # 이전 출력 지움
        lines = input_area.value.splitlines()  # 입력을 줄 단위로 분리
        with output_area:
            for line in lines:
                if line.strip() == '':
                    continue  # 빈 줄 무시
                res = calc(line.strip())  # 각 줄 계산
                history.append(f"{line} = {res}")  # 이력 저장
                print(f"> {line}\n= {res}\n")  # 결과 출력
    calc_btn.on_click(on_calc_click)

    # 위젯을 화면에 디스플레이
    display(VBox([input_area, calc_btn, output_area]))
    # 지원하는 기능 안내 출력
    print("지원: + - * / ^, %, mod, |x|, n!, ln, log(a,b), π, i, sqrt, 삼각/역삼각 함수")
    return True

# calculator_pc: 로컬 PC에서 tkinter로 GUI 계산기 실행
def calculator_pc():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()  # 메인 윈도우 생성
    root.title("실시간 계산기 (PC)")

    history_var = []  # 입력-출력 이력 저장

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    # 입력용 멀티라인 텍스트 위젯
    input_area = tk.Text(frame, height=15, width=40, font=("Arial",14))
    input_area.grid(row=0, column=0, padx=5, pady=5)
    # 출력용 텍스트 위젯(읽기 전용으로 설정)
    output_area = tk.Text(frame, height=15, width=40, font=("Arial",12), state='disabled', fg='blue')
    output_area.grid(row=0, column=1, padx=5, pady=5)

    # safe_dict에 정의된 callable 목록을 콤보박스로 제공
    dropdown = ttk.Combobox(root, values=sorted([k for k in safe_dict.keys() if callable(safe_dict[k])]))
    dropdown.pack(padx=10, pady=5)
    # 콤보 선택 시 입력창에 함수명과 '(' 자동 삽입
    dropdown.bind("<<ComboboxSelected>>", lambda e: input_area.insert("end", dropdown.get()+"("))

    delay_ms = 100  # 키 입력 후 계산까지의 지연 시간(밀리초)
    after_id = None  # 예약된 콜백 ID 저장 (디바운스용)

    # compute_pc: 입력된 각 줄을 계산하여 출력 영역에 결과를 채움
    def compute_pc():
        output_area.config(state='normal')  # 쓰기 가능 상태로 변경
        output_area.delete("1.0", "end")  # 기존 내용 삭제
        lines = input_area.get("1.0", "end").strip().splitlines()  # 줄 단위로 분리
        for line in lines:
            if line.strip() == '':
                continue
            res = calc(line.strip())  # 계산 수행
            history_var.append(f"{line} = {res}")  # 이력에 저장
            output_area.insert("end", f"> {line}\n= {res}\n\n")  # 결과 포맷으로 삽입
        output_area.config(state='disabled')  # 다시 읽기 전용으로 설정

    # schedule_compute: 사용자가 타이핑을 멈추면 일정 시간 후 계산을 수행(디바운스)
    def schedule_compute(event=None):
        nonlocal after_id
        if after_id:
            root.after_cancel(after_id)  # 이전 예약 취소
        after_id = root.after(delay_ms, compute_pc)  # delay_ms 후에 compute_pc 호출 예약

    # 키 릴리즈 이벤트에 디바운스 스케줄러 바인딩
    input_area.bind("<KeyRelease>", schedule_compute)

    # show_history: 별도 창에 계산 이력 출력
    def show_history():
        hist_win = tk.Toplevel(root)
        hist_win.title("이전 기록")
        hist_text = tk.Text(hist_win, height=20, width=60, font=("Arial",12))
        hist_text.pack(padx=10, pady=10)
        hist_text.insert("1.0", "\n".join(history_var))  # 이력을 한 번에 삽입
        hist_text.config(state="disabled")

    hist_btn = ttk.Button(root, text="이전 기록 보기", command=show_history)
    hist_btn.pack(padx=10, pady=5)

    help_label = tk.Label(root, text="지원: + - * / ^, %, mod, |x|, n!, ln, log(a,b), π, i, sqrt, 삼각/역삼각 함수",
                          font=("Arial",10))
    help_label.pack(padx=10, pady=5)

    root.mainloop()  # GUI 이벤트 루프 시작

# main_calculator: 콜랩이면 콜랩 위젯, 아니면 로컬 GUI 실행
def main_calculator():
    """계산기 실행"""
    if not init_calculator_colab():
        calculator_pc()

# ------------------------------
# 기능: 이미지 푸리에 근사
# ------------------------------

def fourier_Transform():

    # download_image: 주어진 URL에서 이미지를 HTTP GET으로 받아 OpenCV 포맷으로 디코딩
    def download_image(url):
        r = requests.get(url)  # HTTP GET 수행
        if r.status_code != 200:
            # 실패 시 상태코드와 함께 예외 발생
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        arr = np.frombuffer(r.content, dtype=np.uint8)  # 바이트 스트림을 uint8 배열로 변환
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함해서 디코딩
        if img is None:
            raise RuntimeError("이미지를 디코딩하지 못했습니다.")
        return img

    # preprocess_image: 다양한 이미지 포맷(BGRA/BGR/GRAY)을 BGR과 이진 마스크로 정규화
    def preprocess_image(img, white_thresh=234):
        """RGBA/GRAY/BGR 모두 안전하게 처리. 반환: (mask, img_bgr)"""
        h, w = img.shape[:2]  # 높이(h)와 너비(w)

        if img.ndim == 3 and img.shape[2] == 4:  # BGRA인 경우 (알파 채널 존재)
            alpha = img[:, :, 3]  # 알파 채널 분리
            alpha_mask = (alpha > 0).astype(np.uint8) * 255  # 알파>0 픽셀만 전경으로
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 색 정보만 BGR로 변환
        elif img.ndim == 3 and img.shape[2] == 3:  # BGR 컬러 이미지
            alpha_mask = np.ones((h, w), dtype=np.uint8) * 255  # 모든 픽셀을 전경 후보로
            img_bgr = img.copy()
        elif img.ndim == 2:  # 그레이스케일 이미지
            alpha_mask = np.ones((h, w), dtype=np.uint8) * 255
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # GRAY -> BGR
        else:
            raise RuntimeError("지원하지 않는 이미지 형식입니다.")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # BGR -> 그레이스케일
        # 밝은 배경(white_thresh 이상)을 배경으로 취급하여 반전 이진화(전경 255)
        _, mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask, alpha_mask)  # 투명 영역(알파) 제거

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # 형태학적 연산용 커널
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 작은 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # 잡음 제거

        return mask, img_bgr

    # find_main_contours: 마스크에서 외곽 컨투어들을 찾아 길이 기준으로 필터링/정렬
    def find_main_contours(mask, min_length=10):
        res = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(res) == 3:
            _, cnts, _ = res  # OpenCV 버전 차이 처리
        else:
            cnts, _ = res
        contours_out = []
        for c in cnts:
            if c is None or c.shape[0] < min_length:
                continue  # 너무 짧으면 잡음으로 간주
            pts = c[:,0,:].astype(float)  # (N,1,2) -> (N,2) 형태로 변환
            contours_out.append(pts)
        # 점 수가 많은 컨투어 순으로 정렬
        contours_out.sort(key=lambda p: p.shape[0], reverse=True)
        return contours_out

    # join_contours_by_nearest: 여러 컨투어를 최근접 끝점 기준으로 차례로 연결
    def join_contours_by_nearest(contours):
        if not contours:
            return np.empty((0,2))  # 빈 경우 빈 배열 반환
        used = [False]*len(contours)  # 각 컨투어의 사용 여부 추적
        idx0 = 0  # 시작 컨투어 인덱스 (첫 번째)
        path = contours[idx0].tolist()  # 시작 컨투어의 점들을 리스트로 변환하여 path 초기화
        used[idx0] = True

        def endpoints(pts):
            return np.array(pts[0]), np.array(pts[-1])  # 컨투어의 시작점과 끝점 반환

        while not all(used):
            last = np.array(path[-1])  # 현재 path의 마지막 점
            best_dist = float('inf')
            best_idx, best_reverse = None, False
            # 아직 사용되지 않은 컨투어들 중 마지막 점과 가장 가까운 끝점을 가진 컨투어 선택
            for i, c in enumerate(contours):
                if used[i]: continue
                c0, c1 = endpoints(c)
                d0 = np.linalg.norm(last - c0)  # 마지막 점과 컨투어의 시작점 거리
                d1 = np.linalg.norm(last - c1)  # 마지막 점과 컨투어의 끝점 거리
                if d0 < best_dist:
                    best_dist = d0; best_idx = i; best_reverse = False
                if d1 < best_dist:
                    best_dist = d1; best_idx = i; best_reverse = True
            if best_idx is None:
                break  # 더 이상 연결 후보가 없으면 종료
            chosen = contours[best_idx]
            if best_reverse:
                chosen = chosen[::-1]  # 필요 시 컨투어를 역순으로 뒤집어 연결 방향 통일
            path.extend(chosen.tolist())  # 선택된 컨투어의 점들을 path에 덧붙임
            used[best_idx] = True
        return np.array(path)  # 최종 연결된 경로 반환

    # resample_path: path를 호 길이(arc length) 기준으로 균일하게 M개의 점으로 재샘플
    def resample_path(path, M=2000):
        if path.shape[0] < 2:
            return path  # 점이 부족하면 원본 반환
        diffs = np.diff(path, axis=0)  # 인접 점 차분
        seglen = np.sqrt((diffs**2).sum(axis=1))  # 각 세그먼트의 길이
        cum = np.concatenate(([0.0], np.cumsum(seglen)))  # 누적 거리 배열 (첫 원소는 0)
        total = cum[-1]  # 전체 경로 길이
        if total == 0:
            # 모든 점이 동일하면 첫 점을 M번 반복해서 반환
            return np.repeat(path[0:1], M, axis=0)
        t_uniform = np.linspace(0, total, M)  # 균일한 누적 거리 값 생성
        resampled = np.empty((M,2))  # 결과 저장용 배열
        resampled[:,0] = np.interp(t_uniform, cum, path[:,0])  # x 좌표 보간
        resampled[:,1] = np.interp(t_uniform, cum, path[:,1])  # y 좌표 보간
        return resampled

    # fourier_from_path: path를 (x(t), y(t))의 사인/코사인 급수로 근사하는 계수 계산
    def fourier_from_path(path, N_terms=30):
        n = path.shape[0]  # 샘플 수
        t = np.linspace(0, 1, n, endpoint=False)  # 0..1 범위의 균일한 t 샘플
        x = path[:,0]; y = path[:,1]  # x, y 좌표 배열
        a0x, a0y = np.mean(x), np.mean(y)  # DC 성분(평균값)
        ax, bx, ay, by = [], [], [], []  # 계수 리스트 초기화
        for k in range(1, N_terms+1):
            cosk = np.cos(2*np.pi*k*t)  # cos(2πk t) 벡터
            sink = np.sin(2*np.pi*k*t)  # sin(2πk t) 벡터
            a_k = 2.0/n * np.sum(x * cosk)  # x의 코사인 계수
            b_k = 2.0/n * np.sum(x * sink)  # x의 사인 계수
            c_k = 2.0/n * np.sum(y * cosk)  # y의 코사인 계수
            d_k = 2.0/n * np.sum(y * sink)  # y의 사인 계수
            ax.append(a_k); bx.append(b_k); ay.append(c_k); by.append(d_k)
        # build: 계수들을 사람이 읽을 수 있는 파라메트릭 수식 문자열로 조합
        def build(a0, a_list, b_list):
            s = f"{a0:.9f}"  # DC 성분 문자열
            for i, (aa, bb) in enumerate(zip(a_list, b_list), start=1):
                s += f" + ({aa:.9f})*cos(2*pi*{i}*t) + ({bb:.9f})*sin(2*pi*{i}*t)"
            s += " {0 <= t <= 1}"  # Desmos 사용을 위한 범위 주석
            return s
        x_expr = build(a0x, ax, bx)  # x(t) 문자열 생성
        y_expr = build(a0y, ay, by)  # y(t) 문자열 생성
        return x_expr, y_expr

    # image_to_fourier_equation: 전체 파이프라인(다운로드→전처리→컨투어→연결→재샘플→푸리에) 수행
    def image_to_fourier_equation(url, N_terms=40, resample_M=2000, preview=True, white_thresh=234):
        img = download_image(url)  # URL에서 이미지 가져오기
        mask, img_bgr = preprocess_image(img, white_thresh=white_thresh)  # 마스크 및 BGR 이미지 취득
        contours = find_main_contours(mask, min_length=5) # 컨투어 추출 (min_length로 잡음 제거)
        if len(contours) == 0:
            # 컨투어가 없으면 예외로 처리하여 사용자에게 알림
            raise RuntimeError("컨투어를 찾지 못했습니다. 임계치(white_thresh)를 낮춰보세요.")
        if len(contours) == 1:
            path = contours[0]  # 하나이면 그대로 사용
        else:
            path = join_contours_by_nearest(contours)  # 여러개이면 연결하여 하나의 경로로 만듦

        path_rs = resample_path(path, M=resample_M)  # 재샘플링 수행
        path_rs[:,1] = -path_rs[:,1]  # 화면 좌표계(y 아래가 증가) → 수학 좌표계로 변환(상하 반전)

        if preview:
            # 중간 결과(마스크, 컨투어, 재샘플 경로)를 시각화
            plt.figure(figsize=(8,8))
            plt.subplot(1,2,1)
            plt.title("Binary mask with contours")
            plt.imshow(mask, cmap='gray')
            for c in contours[:5]:
                plt.plot(c[:,0], c[:,1], linewidth=0.5)
            plt.axis('equal')
            plt.subplot(1,2,2)
            plt.title("Ordered + Resampled Path (Flipped)")
            plt.plot(path_rs[:,0], path_rs[:,1], '-', linewidth=0.5)
            plt.axis('equal')
            plt.show()

        x_expr, y_expr = fourier_from_path(path_rs, N_terms=N_terms)  # 푸리에 식 계산
        print("\n=== DESMOS PARAMETRIC EQUATIONS ===\n")
        print("x(t) =", x_expr)  # x(t) 출력
        print("y(t) =", y_expr)  # y(t) 출력
        return x_expr, y_expr, path_rs

    # fourier_Transform이 스크립트로 직접 실행될 때의 상호작용 코드
    if __name__ == "__main__":
        test_url = str(input("input image url : "))  # 이미지 URL 입력
        xexpr, yexpr, path = image_to_fourier_equation(test_url, N_terms=30, resample_M=2000, preview=True)
        '''
        N_terms 는 n차 푸리에 근사, research_M 은 해상도
        '''
        print("=============Image to Fourier Equation=============")

        # make_func: Desmos 스타일의 문자열을 numpy 기반 수식으로 바꿔 eval 가능한 함수로 생성
        def make_func(expr):
            """문자열 방정식을 t값 배열로 계산할 수 있는 함수로 변환"""
            expr = expr.replace("{0 <= t <= 1}", "")
            expr = expr.replace("cos", "np.cos")
            expr = expr.replace("sin", "np.sin")
            expr = expr.replace("pi", "np.pi")
            def f(t):
                return eval(expr)  # t는 numpy 배열인 경우 배열 연산으로 결과 얻음
            return f

        x_func = make_func(xexpr)  # x(t) 계산 함수 생성
        y_func = make_func(yexpr)  # y(t) 계산 함수 생성

        # === 2. t를 0~1 범위에서 샘플링 ===
        t = np.linspace(0, 1, 10000)  # 해상도 높은 샘플링

        # === 3. x(t), y(t) 계산 ===
        x = x_func(t)  # x(t) 값 배열
        y = y_func(t)  # y(t) 값 배열

        # === 4. 그래프 그리기 ===
        plt.figure(figsize=(8, 8))
        plt.plot(x, y, linewidth=0.8)  # 근사 곡선 플로팅
        plt.title("Fourier Approximation")
        plt.axis("equal")  # x,y 비율 동일하게 설정
        plt.show()
    return 0

# ------------------------------
# 메인 메뉴: 사용자가 실행할 기능을 선택하도록 함
# ------------------------------

def main():
    answer = input(
        "=============================\n어떤 프로그램을 실행하시겠습니까? \n 1. 소인수 분해\n 2. 난수 생성 \n 3. 이미지 푸리에 근사 \n 4. GUI 포함 계산기\n그만 입력하면 종료\n=============================\n")
    if answer in ["4", "계산기"]:
        main_calculator()

    else :
        while True:
            if answer in ["그만", "stop", ""]:
                print("프로그램을 종료합니다.")
                break

            elif answer in ["1", "소인수분해"]:
                print("소인수분해 기능 실행 (예시)")
                fractorization()

            elif answer in ["2", "난수 생성"]:
                print("난수 생성 기능 실행 (예시)")
                # random_gen() 호출

            elif answer in ["3", "이미지 푸리에 근사"]:
                print("이미지 푸리에 기능 실행 (예시)")
                fourier_Transform()


            else:
                print("알 수 없는 입력입니다. 다시 시도하세요.")

if __name__ == "__main__":
    main()



---
---
---
# Variables
---
변수명,역할
cv2,OpenCV 모듈: 이미지 읽기/처리
np,NumPy 모듈: 수치 연산 및 배열 처리
requests,HTTP 요청을 통해 이미지/데이터 다운로드
plt,Matplotlib 모듈: 그래프/이미지 시각화
math,표준 수학 함수(제곱근, 로그, 삼각함수 등)
cmath,복소수 연산용 함수
re,정규표현식: 문자열 전처리
sys,시스템 관련 유틸
safe_dict,eval 안전용 허용 이름 사전
expr,계산기 입력 문자열
lines,계산기 입력 문자열을 줄 단위로 분리한 리스트
res,계산 결과
history,계산 이력 리스트(Colab)
input_area,입력용 위젯(Textarea or Text)
output_area,출력용 위젯(Textarea or Text)
calc_btn,계산 버튼 위젯
history_var,계산 이력 리스트(PC)
root,tkinter 메인 윈도우
frame,tkinter 프레임
dropdown,tkinter 콤보박스
delay_ms,키 입력 후 계산 지연(ms)
after_id,예약된 콜백 ID
put,사용자 입력 소인수분해할 정수
n,소인수분해 과정에서 현재 나머지
primelist,발견된 소인수 리스트
powerlist,각 소인수의 지수 리스트
p,현재 추출된 소인수
e,현재 추출된 소인수의 지수
img,다운로드된 이미지 배열
mask,전경 마스크
img_bgr,BGR 이미지
alpha,알파 채널
alpha_mask,알파 채널 기반 마스크
gray,그레이스케일 이미지
kernel,모폴로지 연산 커널
cnts,발견된 모든 컨투어
contours_out,길이 기준 필터링된 컨투어 리스트
used,컨투어 사용 여부 리스트
idx0,초기 컨투어 인덱스
path,연결된 경로 또는 재샘플 경로
last,현재 path의 마지막 점
best_dist,최근접 컨투어 거리
best_idx,선택된 컨투어 인덱스
best_reverse,컨투어 뒤집기 여부
chosen,선택된 컨투어
diffs,인접 점 차분
seglen,각 세그먼트 길이
cum,누적 거리 배열
total,전체 경로 길이
t_uniform,균일 누적 거리 값
resampled,재샘플링된 경로
x,x좌표 배열 또는 함수 결과
y,y좌표 배열 또는 함수 결과
a0x,x DC 성분
a0y,y DC 성분
ax,x 코사인 계수 리스트
bx,x 사인 계수 리스트
ay,y 코사인 계수 리스트
by,y 사인 계수 리스트
k,주파수 인덱스 반복 변수
cosk,cos(2πk t) 벡터
sink,sin(2πk t) 벡터
a_k,x 코사인 계수
b_k,x 사인 계수
c_k,y 코사인 계수
d_k,y 사인 계수
s,문자열로 조합된 수식
x_expr,Desmos 스타일 x(t) 수식
y_expr,Desmos 스타일 y(t) 수식
preview,중간 결과 시각화 여부
N_terms,푸리에 항 개수
resample_M,재샘플 샘플 수
test_url,테스트용 이미지 URL
x_func,x(t) 계산 함수
y_func,y(t) 계산 함수
t,0~1 범위 샘플링 값 배열


---
---
---
# Presentation
프로그램 목적:

수학/이미지/계산 기능을 하나로 통합한 파이썬 툴

주요 기능:

소인수분해

사용자가 입력한 정수를 소인수와 지수 형태로 출력

예: 360 → 2^3 x 3^2 x 5^1

난수 생성

(코드 예시에서는 placeholder, 실제 난수 생성 가능)

이미지 푸리에 근사

이미지 URL 입력 → OpenCV로 이미지 읽기

전처리: 배경 제거, 컨투어 추출, 경로 재샘플링

푸리에 근사: x(t), y(t) 파라메트릭 방정식 생성 → Desmos/Matplotlib 출력

시각화: 근사 곡선을 그래프로 확인 가능

GUI 계산기

Colab 환경: ipywidgets 기반 실시간 계산기

로컬 PC: Tkinter 기반 실시간 계산기

지원 연산: +, -, *, /, ^, %, mod, n!, ln, log(a,b), π, i, sqrt, 삼각/역삼각 함수

프로그램 구조:

main() → 사용자 선택 기반 기능 실행

안전한 eval() 환경(safe_dict) → 문자열 수식 계산

이미지 푸리에 근사는 함수형 파이프라인:

다운로드 → 2. 전처리 → 3. 컨투어 추출 → 4. 경로 연결 → 5. 재샘플링 → 6. 푸리에 근사 → 7. 시각화

사용 예시:

소인수분해: 1 입력 → 프로그램 출력

이미지 푸리에: URL 입력 → x(t), y(t) 출력 + 플롯

계산기: 실시간 수식 입력 → 결과 출력
