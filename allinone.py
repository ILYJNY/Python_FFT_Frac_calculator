import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import math
import cmath
import re
import sys

# ---------------- 기능: 소인수분해 ----------------
def fractorization():
    def pen(n):
        for i in range(2, math.floor(math.sqrt(n)) + 1):
            if n % i == 0:
                for j in range(0, math.floor(math.log(n, i)) + 2):
                    if n % i ** j != 0:
                        return i, j-1, n / (i ** (j-1))
        return n, 1, 1

    put = int(input('소인수분해할 정수 입력: '))
    n = put
    primelist = []
    powerlist = []
    while True:
        p, e, n = pen(n)
        primelist.append(p)
        powerlist.append(e)
        if n == 1:
            break

    print("\n===== 소인수분해 결과 =====")
    if len(primelist) == 1:
        print(f'{primelist[0]}^1 (소수)')
    else:
        for i in range(len(primelist)-1):
            print(f'{primelist[i]}^{powerlist[i]} x', end=' ')
        print(f'{int(primelist[-1])}^{powerlist[-1]} (소수가 아님)')

# ---------------- 기능: 계산기 (Colab/PC) ----------------
safe_dict = {
    'pi': math.pi, 'π': math.pi, 'e': math.e, 'i': 1j, 'j': 1j,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'ln': math.log, 'log': lambda x, base=10: math.log(x, base),
    'sqrt': math.sqrt, 'abs': abs, 'mod': lambda x, y: x%y,
    'gamma': math.gamma, 'exp': math.exp, 'pow': pow
}

def preprocess(expr):
    expr = expr.replace('^','**').replace('%',' mod ').replace('π','pi')
    expr = re.sub(r'(\d+)!', r'gamma(\1+1)', expr)
    expr = re.sub(r'\|([^\|]+)\|', r'abs(\1)', expr)
    return expr

def calc(expr):
    expr = preprocess(expr)
    try:
        return eval(expr, {"__builtins__":None}, safe_dict)
    except Exception as e:
        return f"계산 오류: {e}"

def init_calculator_colab():
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if not IN_COLAB:
        return False

    from ipywidgets import Textarea, Button, Output, VBox
    from IPython.display import display

    input_area = Textarea(placeholder="계산식을 입력하세요...", layout={'width':'50%', 'height':'150px'})
    output_area = Output(layout={'border':'1px solid black', 'width':'50%', 'height':'150px'})
    calc_btn = Button(description="계산")
    history = []

    def on_calc_click(b):
        output_area.clear_output()
        lines = input_area.value.splitlines()
        with output_area:
            for line in lines:
                if line.strip() == '':
                    continue
                res = calc(line.strip())
                history.append(f"{line} = {res}")
                print(f"> {line}\n= {res}\n")
    calc_btn.on_click(on_calc_click)
    display(VBox([input_area, calc_btn, output_area]))
    print("지원: + - * / ^, %, mod, |x|, n!, ln, log(a,b), π, i, sqrt, 삼각/역삼각 함수")
    return True

def calculator_pc():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("실시간 계산기 (PC)")

    history_var = []

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    input_area = tk.Text(frame, height=15, width=40, font=("Arial",14))
    input_area.grid(row=0, column=0, padx=5, pady=5)
    output_area = tk.Text(frame, height=15, width=40, font=("Arial",12), state='disabled', fg='blue')
    output_area.grid(row=0, column=1, padx=5, pady=5)

    dropdown = ttk.Combobox(root, values=sorted([k for k in safe_dict.keys() if callable(safe_dict[k])]))
    dropdown.pack(padx=10, pady=5)
    dropdown.bind("<<ComboboxSelected>>", lambda e: input_area.insert("end", dropdown.get()+"("))

    delay_ms = 100
    after_id = None

    def compute_pc():
        output_area.config(state='normal')
        output_area.delete("1.0", "end")
        lines = input_area.get("1.0", "end").strip().splitlines()
        for line in lines:
            if line.strip() == '':
                continue
            res = calc(line.strip())
            history_var.append(f"{line} = {res}")
            output_area.insert("end", f"> {line}\n= {res}\n\n")
        output_area.config(state='disabled')

    def schedule_compute(event=None):
        nonlocal after_id
        if after_id:
            root.after_cancel(after_id)
        after_id = root.after(delay_ms, compute_pc)

    input_area.bind("<KeyRelease>", schedule_compute)

    def show_history():
        hist_win = tk.Toplevel(root)
        hist_win.title("이전 기록")
        hist_text = tk.Text(hist_win, height=20, width=60, font=("Arial",12))
        hist_text.pack(padx=10, pady=10)
        hist_text.insert("1.0", "\n".join(history_var))
        hist_text.config(state="disabled")

    hist_btn = ttk.Button(root, text="이전 기록 보기", command=show_history)
    hist_btn.pack(padx=10, pady=5)

    help_label = tk.Label(root, text="지원: + - * / ^, %, mod, |x|, n!, ln, log(a,b), π, i, sqrt, 삼각/역삼각 함수",
                          font=("Arial",10))
    help_label.pack(padx=10, pady=5)

    root.mainloop()

def main_calculator():
    """계산기 실행"""
    if not init_calculator_colab():
        calculator_pc()

# ------------------------------
# 기능: 이미지 푸리에 근사
# ------------------------------

def fourier_Transform():


    # if __name__ == "__main__":
    #     main()
    def download_image(url):
        r = requests.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("이미지를 디코딩하지 못했습니다.")
        return img

    def preprocess_image(img, white_thresh=234):
        """
        RGBA/GRAY/BGR 모두 안전하게 처리.
        - 반환: binary mask (0/255), BGR 이미지
        """
        h, w = img.shape[:2]

        if img.ndim == 3 and img.shape[2] == 4:  # BGRA
            alpha = img[:, :, 3]
            alpha_mask = (alpha > 0).astype(np.uint8) * 255
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:  # BGR
            alpha_mask = np.ones((h, w), dtype=np.uint8) * 255
            img_bgr = img.copy()
        elif img.ndim == 2:  # GRAY
            alpha_mask = np.ones((h, w), dtype=np.uint8) * 255
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise RuntimeError("지원하지 않는 이미지 형식입니다.")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask, alpha_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask, img_bgr

    def find_main_contours(mask, min_length=10):
        res = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(res) == 3:
            _, cnts, _ = res
        else:
            cnts, _ = res
        contours_out = []
        for c in cnts:
            if c is None or c.shape[0] < min_length:
                continue
            pts = c[:,0,:].astype(float)
            contours_out.append(pts)
        contours_out.sort(key=lambda p: p.shape[0], reverse=True)
        return contours_out

    def join_contours_by_nearest(contours):
        if not contours:
            return np.empty((0,2))
        used = [False]*len(contours)
        idx0 = 0
        path = contours[idx0].tolist()
        used[idx0] = True

        def endpoints(pts):
            return np.array(pts[0]), np.array(pts[-1])

        while not all(used):
            last = np.array(path[-1])
            best_dist = float('inf')
            best_idx, best_reverse = None, False
            for i, c in enumerate(contours):
                if used[i]: continue
                c0, c1 = endpoints(c)
                d0 = np.linalg.norm(last - c0)
                d1 = np.linalg.norm(last - c1)
                if d0 < best_dist:
                    best_dist = d0; best_idx = i; best_reverse = False
                if d1 < best_dist:
                    best_dist = d1; best_idx = i; best_reverse = True
            if best_idx is None:
                break
            chosen = contours[best_idx]
            if best_reverse:
                chosen = chosen[::-1]
            path.extend(chosen.tolist())
            used[best_idx] = True
        return np.array(path)

    def resample_path(path, M=2000):
        if path.shape[0] < 2:
            return path
        diffs = np.diff(path, axis=0)
        seglen = np.sqrt((diffs**2).sum(axis=1))
        cum = np.concatenate(([0.0], np.cumsum(seglen)))
        total = cum[-1]
        if total == 0:
            return np.repeat(path[0:1], M, axis=0)
        t_uniform = np.linspace(0, total, M)
        resampled = np.empty((M,2))
        resampled[:,0] = np.interp(t_uniform, cum, path[:,0])
        resampled[:,1] = np.interp(t_uniform, cum, path[:,1])
        return resampled

    def fourier_from_path(path, N_terms=30):
        n = path.shape[0]
        t = np.linspace(0, 1, n, endpoint=False)
        x = path[:,0]; y = path[:,1]
        a0x, a0y = np.mean(x), np.mean(y)
        ax, bx, ay, by = [], [], [], []
        for k in range(1, N_terms+1):
            cosk = np.cos(2*np.pi*k*t)
            sink = np.sin(2*np.pi*k*t)
            a_k = 2.0/n * np.sum(x * cosk)
            b_k = 2.0/n * np.sum(x * sink)
            c_k = 2.0/n * np.sum(y * cosk)
            d_k = 2.0/n * np.sum(y * sink)
            ax.append(a_k); bx.append(b_k); ay.append(c_k); by.append(d_k)
        def build(a0, a_list, b_list):
            s = f"{a0:.9f}"
            for i, (aa, bb) in enumerate(zip(a_list, b_list), start=1):
                s += f" + ({aa:.9f})*cos(2*pi*{i}*t) + ({bb:.9f})*sin(2*pi*{i}*t)"
            s += " {0 <= t <= 1}"
            return s
        x_expr = build(a0x, ax, bx)
        y_expr = build(a0y, ay, by)
        return x_expr, y_expr

    def image_to_fourier_equation(url, N_terms=40, resample_M=2000, preview=True, white_thresh=234):
        img = download_image(url)
        mask, img_bgr = preprocess_image(img, white_thresh=white_thresh)
        contours = find_main_contours(mask, min_length=5) # min_length는 얼마나 세밀하게 연결할건지
        if len(contours) == 0:
            raise RuntimeError("컨투어를 찾지 못했습니다. 임계치(white_thresh)를 낮춰보세요.")
        if len(contours) == 1:
            path = contours[0]
        else:
            path = join_contours_by_nearest(contours)

        path_rs = resample_path(path, M=resample_M)
        path_rs[:,1] = -path_rs[:,1]  # ✅ 상하 반전 보정 추가

        if preview:
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

        x_expr, y_expr = fourier_from_path(path_rs, N_terms=N_terms)
        print("\n=== DESMOS PARAMETRIC EQUATIONS ===\n")
        print("x(t) =", x_expr)
        print("y(t) =", y_expr)
        return x_expr, y_expr, path_rs

    if __name__ == "__main__":
        test_url = str(input("input image url : "))
        xexpr, yexpr, path = image_to_fourier_equation(test_url, N_terms=30, resample_M=2000, preview=True)
        '''
        N_terms 는 n차 푸리에 근사, research_M 은 해상도
        '''
        print("=============Image to Fourier Equation=============")

        def make_func(expr):
            """문자열 방정식을 t값 배열로 계산할 수 있는 함수로 변환"""
            expr = expr.replace("{0 <= t <= 1}", "")
            expr = expr.replace("cos", "np.cos")
            expr = expr.replace("sin", "np.sin")
            expr = expr.replace("pi", "np.pi")
            def f(t):
                return eval(expr)



            return f

        x_func = make_func(xexpr)
        y_func = make_func(yexpr)

        # === 2. t를 0~1 범위에서 샘플링 ===
        t = np.linspace(0, 1, 10000)

        # === 3. x(t), y(t) 계산 ===
        x = x_func(t)
        y = y_func(t)

        # === 4. 그래프 그리기 ===
        plt.figure(figsize=(8, 8))
        plt.plot(x, y, linewidth=0.8)
        plt.title("Fourier Approximation")
        plt.axis("equal")  # x, y 비율 동일하게
        plt.show()
    return 0

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