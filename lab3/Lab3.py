import cv2
import numpy as np
from collections import defaultdict

def get_otsu(gray):
    min_within_var = float('inf')
    best_t = 0

    for t in range(256):
        group1 = gray[gray <= t]
        group2 = gray[gray > t]
        if len(group1) == 0 or len(group2) == 0:
            continue
        var1 = np.var(group1)
        var2 = np.var(group2)
        within_var = var1 + var2
        if within_var < min_within_var:
            min_within_var = within_var
            best_t = t

    otsu_result = np.zeros_like(gray)
    otsu_result[gray > best_t] = 255
    return otsu_result

# ============================================================
# Two-Pass Connected Component Labeling 
# ============================================================
def two_pass_label(binary: np.ndarray, connectivity: int = 8):
    """
    binary: uint8 二值影像，前景=255，背景=0
    connectivity: 4 或 8
    return:
      labels: int32，每個前景像素都有一個標籤
      stats:  各標籤的統計資訊（面積 + 外接框）
    """
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)

    parent = [0]  # 0 給背景
    rank = [0]

    def make_set():
        parent.append(len(parent))
        rank.append(0)
        return len(parent) - 1

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if rank[ra] < rank[rb]:
            parent[ra] = rb
            return rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
            return ra
        else:
            parent[rb] = ra
            rank[ra] += 1
            return ra

    neighbors = [(-1,0),(0,-1)] if connectivity==4 else [(-1,-1),(-1,0),(-1,1),(0,-1)]

    # -------- Pass 1 --------
    for y in range(H):
        for x in range(W):
            if binary[y,x] == 0:
                continue
            cur_label = make_set()
            labels[y,x] = cur_label
            for dy,dx in neighbors:
                ny, nx = y+dy, x+dx
                if 0<=ny<H and 0<=nx<W and binary[ny,nx]!=0:
                    union(cur_label, labels[ny,nx])

    # -------- Pass 2 --------
    label_map = {0:0}
    next_id = 1
    stats = defaultdict(lambda: {'area':0,'minx':10**9,'miny':10**9,'maxx':-10**9,'maxy':-10**9})

    for y in range(H):
        for x in range(W):
            lab = labels[y,x]
            if lab == 0:
                continue
            root = find(lab)
            if root not in label_map:
                label_map[root] = next_id
                next_id += 1
            new_id = label_map[root]
            labels[y,x] = new_id

            st = stats[new_id]
            st['area'] += 1
            st['minx'] = min(st['minx'], x)
            st['miny'] = min(st['miny'], y)
            st['maxx'] = max(st['maxx'], x)
            st['maxy'] = max(st['maxy'], y)

    return labels, stats

# ============================================================
# 第一題：C
# ============================================================
def connected_component(image_path: str, out_path: str = "cc_result.png", connectivity: int = 8):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)

    otsu = get_otsu(gray)
    labels, stats = two_pass_label(otsu, connectivity)

    num_labels = labels.max()
    palette = np.zeros((num_labels + 1, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for k in range(1, num_labels + 1):
        palette[k] = rng.integers(0, 256, size=3, dtype=np.uint8)
    color_out = palette[labels]

    cv2.imwrite(out_path, color_out)
    print(f"[Connected Component] 找到 {num_labels} 個區域, 已存檔 {out_path}")
# ============================================================
# 第二題：
# ============================================================
def foreground_detection(video_path: str, out_path: str = "output.mp4", area_threshold: int = 250, connectivity: int = 8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    # 取得影片參數
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Background Subtractor (調整參數提高穩定度)
    backSub = cv2.createBackgroundSubtractorMOG2(
        varThreshold=20,   # 靈敏度（越小越容易抓到變化）
        detectShadows=True
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = backSub.apply(frame)

        shadow_val = backSub.getShadowValue()
        ret, nmask = cv2.threshold(fgmask, shadow_val, 255, cv2.THRESH_BINARY)

        labels, stats = two_pass_label(nmask, connectivity)

        vis = frame.copy()
        for lid, st in stats.items():
            w = st['maxx'] - st['minx']
            h = st['maxy'] - st['miny']
            if st['area'] < area_threshold:
                continue
            cv2.rectangle(vis, (st['minx'], st['miny']), (st['maxx'], st['maxy']), (0, 255, 0), 2)
        out.write(vis)

        # 如果要同時顯示可以打開這個
        cv2.imshow("Detections", vis)
        if cv2.waitKey(30) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[Foreground Detection] 已輸出結果影片: {out_path}")


# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":
    connected_component("connected_component.jpg", out_path="cc_result.png")
    foreground_detection("car.mp4", area_threshold=200)
