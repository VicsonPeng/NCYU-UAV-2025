# -*- coding: utf-8 -*-
"""
Lab3 作業程式
1. Connected Component (圖片) - 手刻 Two-Pass 演算法
2. Foreground Detection (影片) - Background Subtraction + Connected Component
"""

import cv2
import numpy as np
from collections import defaultdict

# ============================================================
# Union-Find (Disjoint Set) 資料結構
# 用來處理「等價關係」，例如當上、左兩個標籤不同時，
# 我們要把它們視為同一個連通元件，這裡用 Union-Find 來記錄關係。
# ============================================================
class UnionFind:
    def __init__(self):
        self.parent = [0]  # parent[0] = 0，保留給背景
        self.rank = [0]

    def make_set(self):
        """建立新集合（新標籤）"""
        self.parent.append(len(self.parent))
        self.rank.append(0)
        return len(self.parent) - 1

    def find(self, x):
        """尋找根節點，並做路徑壓縮"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        """合併兩個集合（等價關係）"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
            return rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
            return ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
            return ra

# ============================================================
# Two-Pass Connected Component Labeling (手刻版本)
# Step 1: Pass1 - 為每個前景像素賦予暫時標籤，並建立等價關係
# Step 2: Pass2 - 用 Union-Find 解開等價關係，並重新壓縮成 1..K 的標籤
# ============================================================
def two_pass_label(binary: np.ndarray, connectivity: int = 8):
    """
    binary: uint8 二值影像，前景=255，背景=0
    connectivity: 4 或 8，代表 4-連通或 8-連通
    return:
      labels: int32，每個前景像素都有一個標籤
      stats:  各標籤的統計資訊（面積 + 外接框）
    """
    assert connectivity in (4, 8)
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)  # 標籤影像
    uf = UnionFind()  # 等價關係結構

    # 定義鄰居（只看「已經掃過」的像素）
    if connectivity == 4:
        neighbors = [(-1, 0), (0, -1)]  # 上、左
    else:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]  # 左上、上、右上、左

    # -------- Pass 1：逐像素掃描 --------
    for y in range(H):
        for x in range(W):
            if binary[y, x] == 0:
                continue  # 背景不標記

            # 先給一個新的暫時標籤
            cur_label = uf.make_set()
            labels[y, x] = cur_label

            # 檢查鄰居是否有已標記的像素
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] != 0:
                    nbr_lab = labels[ny, nx]
                    if nbr_lab > 0:
                        uf.union(cur_label, nbr_lab)  # 建立等價關係

    # -------- Pass 2：解開等價關係並重新編號 --------
    label_map = {0: 0}  # 背景維持 0
    next_id = 1
    used = np.unique(labels)
    for lab in used:
        if lab == 0:
            continue
        root = uf.find(lab)  # 找出這個標籤的根
        if root not in label_map:
            label_map[root] = next_id
            next_id += 1

    # 建立統計資訊
    stats = defaultdict(lambda: {
        'area': 0, 'minx': 10**9, 'miny': 10**9,
        'maxx': -10**9, 'maxy': -10**9
    })

    for y in range(H):
        for x in range(W):
            lab = labels[y, x]
            if lab == 0:
                continue
            new_id = label_map[uf.find(lab)]  # 將暫時標籤轉換為新 ID
            labels[y, x] = new_id

            st = stats[new_id]
            st['area'] += 1
            st['minx'] = min(st['minx'], x)
            st['miny'] = min(st['miny'], y)
            st['maxx'] = max(st['maxx'], x)
            st['maxy'] = max(st['maxy'], y)

    return labels, stats

# ============================================================
# 第一題：Connected Component (圖片)
# ============================================================
def connected_component(image_path: str, out_path: str = "cc_result.png", connectivity: int = 8):
    # 1) 讀取灰階圖
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)

    # 2) Otsu 自動閾值二值化
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) 連通元件標記
    labels, stats = two_pass_label(otsu, connectivity)

    # 4) 給每個元件隨機顏色
    num_labels = labels.max()
    palette = np.zeros((num_labels + 1, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for k in range(1, num_labels + 1):
        palette[k] = rng.integers(0, 256, size=3, dtype=np.uint8)
    color_out = palette[labels]

    # 5) 存檔
    cv2.imwrite(out_path, color_out)
    print(f"[Connected Component] 找到 {num_labels} 個區域, 已存檔 {out_path}")

# ============================================================
# 第二題：Foreground Detection (影片)
# ============================================================
def foreground_detection(video_path: str, area_threshold: int = 3000, connectivity: int = 8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    backSub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = backSub.apply(frame)
        shadow_val = backSub.getShadowValue()
        _, nmask = cv2.threshold(fgmask, shadow_val, 255, cv2.THRESH_BINARY)

        # --- 加上去雜訊處理 ---
        kernel = np.ones((5,5), np.uint8)
        nmask = cv2.morphologyEx(nmask, cv2.MORPH_OPEN, kernel, iterations=2)

        labels, stats = two_pass_label(nmask, connectivity)

        vis = frame.copy()
        for lid, st in stats.items():
            w = st['maxx'] - st['minx']
            h = st['maxy'] - st['miny']
            if st['area'] > area_threshold and w > 30 and h > 30:
                cv2.rectangle(vis, (st['minx'], st['miny']), (st['maxx'], st['maxy']), (0, 255, 0), 2)
                cv2.putText(vis, f"car:{lid}", (st['minx'], max(0, st['miny']-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow("Detections", vis)
        cv2.imshow("Mask", nmask)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 主程式 
# ============================================================
if __name__ == "__main__":
    # ---- 第一題：Connected Component ----
    connected_component("connected_component.jpg", out_path="cc_result.png")

    # ---- 第二題：Foreground Detection ----
    foreground_detection("car.mp4", area_threshold=800)
