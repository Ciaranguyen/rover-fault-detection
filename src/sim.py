"""Mô phỏng đội rover phủ vùng, có định vị tự ước lượng, đo khoảng cách UWB và robot báo sai.

Mỗi episode sinh ra:
    p_true  (T, N, 2)  vị trí thật          -> CHỈ dùng để phân tích, KHÔNG đưa vào mô hình
    p_hat   (T, N, 2)  vị trí mỗi rover tự khai (gửi cho đồng đội)
    r       (T, N, N)  khoảng cách UWB đo được, đối xứng, NaN nếu ngoài tầm hoặc mất gói
    label   (T, N)     0 = trung thực, 1 = trôi định vị (F-drift), 2 = khai giả (A-spoof)
    meta               loại lỗi, rover lỗi, thời điểm lỗi bắt đầu, mức trôi, ...

Các loại lỗi:
    drift     : p_hat lệch dần khỏi p theo quãng đường đi được; rover vẫn làm việc trung thực
    spoof     : từ t_u, rover khai một quỹ đạo giả đi về phía vùng khác
    spoof_one : như spoof, nhưng vị trí giả được chọn để KHỚP range với một rover nhận
                (rover gần nhất lúc t_u); chỉ các rover khác mới thấy sai lệch

Nhiễu đo lấy từ dữ liệu UWB thật (src/noise_tiers.py): độ lệch riêng mỗi cặp + phần dư ngẫu nhiên.
"""
from dataclasses import dataclass, field, asdict

import numpy as np

HONEST, DRIFT, SPOOF = 0, 1, 2


@dataclass
class SimConfig:
    n_rovers: int = 5
    area: float = 24.0            # cạnh vùng vuông (m)
    comm_range: float = 10.0      # tầm UWB (m); dữ liệu nhiễu thật đo ở 0,4–8,7 m
    steps: int = 400              # số bước, dt = 1 s
    speed: float = 0.3            # m/s
    lane_gap: float = 1.5         # khoảng cách giữa các làn quét (m)
    p_drop: float = 0.1           # xác suất mất một phép đo
    # định vị
    honest_drift_max: float = 0.005       # rover trung thực cũng trôi nhẹ: 0–0,5% quãng đường
    drift_rate: tuple = (0.02, 0.05)      # mức trôi của rover lỗi (tỉ lệ quãng đường)
    drift_turn_std: float = 0.05          # độ đổi hướng trôi mỗi bước (rad)
    # lỗi
    p_fault: float = 0.7                  # xác suất episode có rover lỗi
    n_faulty: int = 1
    fault_mix: dict = field(default_factory=lambda: {"drift": 0.5, "spoof": 0.25, "spoof_one": 0.25})
    t_fault: tuple = (100, 250)           # khoảng thời điểm lỗi bắt đầu
    # nhiễu đo
    noise: str = "empirical"              # "empirical" (TIERS) hoặc "gaussian"
    noise_scale: float = 1.0              # nhân phần dư (dùng cho tập ngoài phân phối)
    residual_case: int = 4                # case TIERS để lấy phần dư (4 = robot di chuyển)
    noise_temporal: str = "block"         # "block": ghép các đoạn liên tiếp của chuỗi nhiễu thật
                                          # (giữ tương quan thời gian); "iid": bốc độc lập từng bước
    noise_block: int = 20                 # độ dài mỗi đoạn (bước = giây)
    calibrate_global_bias: bool = True    # trừ độ lệch trung vị chung, như hiệu chỉnh trễ ăng-ten


class NoiseModel:
    def __init__(self, path, cfg: SimConfig):
        z = np.load(path)
        self.bias = z["bias"]
        res = z["residual"]
        if cfg.residual_case is not None:
            res = res[z["source"] == cfg.residual_case]
        self.residual = res
        # chuỗi 1 Hz theo thời gian, để ghép đoạn (block bootstrap)
        off = z["series_offsets"]
        keep = [i for i in range(len(off) - 1)
                if cfg.residual_case is None or z["series_source"][i] == cfg.residual_case]
        self.series = [z["series"][off[i]:off[i + 1]] for i in keep]
        self.global_bias = np.median(self.bias) if cfg.calibrate_global_bias else 0.0
        self.cfg = cfg
        # tham số Gaussian cùng mức độ lệch chuẩn, dùng cho thí nghiệm E6
        self.g_bias_mu, self.g_bias_sd = self.bias.mean(), self.bias.std()
        self.g_res_sd = res.std()

    def pair_bias(self, rng, n_pairs):
        if self.cfg.noise == "gaussian":
            b = rng.normal(self.g_bias_mu, self.g_bias_sd, n_pairs)
        else:
            b = rng.choice(self.bias, n_pairs)
        return b - self.global_bias

    def residuals(self, rng, shape):
        """shape = (T, số cặp)."""
        if self.cfg.noise == "gaussian":
            e = rng.normal(0.0, self.g_res_sd, shape)
        elif self.cfg.noise_temporal == "iid":
            e = rng.choice(self.residual, shape)
        else:
            T, P = shape
            B = self.cfg.noise_block
            e = np.empty(shape, np.float32)
            for p in range(P):
                col, t = [], 0
                while t < T:
                    s = self.series[rng.integers(len(self.series))]
                    L = min(B, len(s), T - t)
                    st = rng.integers(0, len(s) - L + 1)
                    col.append(s[st:st + L])
                    t += L
                e[:, p] = np.concatenate(col)[:T]
        return self.cfg.noise_scale * e


def lawnmower_path(x0, x1, area, gap, speed, steps, rng):
    """Quỹ đạo quét làn trong dải [x0, x1]; quét xong thì đi điểm ngẫu nhiên."""
    xs = np.arange(x0 + gap / 2, x1, gap)
    wps = []
    for i, x in enumerate(xs):
        ys = (0.5, area - 0.5) if i % 2 == 0 else (area - 0.5, 0.5)
        wps += [(x, ys[0]), (x, ys[1])]
    pos = np.array(wps[0], float)
    path, k = [], 1
    for _ in range(steps):
        path.append(pos.copy())
        if k >= len(wps):
            wps.append(tuple(rng.uniform(0.5, area - 0.5, 2)))
        target = np.asarray(wps[k])
        d = target - pos
        dist = np.linalg.norm(d)
        step = speed * rng.uniform(0.8, 1.2)
        if dist <= step:
            pos, k = target.copy(), k + 1
        else:
            pos = pos + d / dist * step
    return np.asarray(path)


def simulate_episode(cfg: SimConfig, noise: NoiseModel, rng):
    N, T, L = cfg.n_rovers, cfg.steps, cfg.area
    order = rng.permutation(N)  # xáo thứ tự rover để mô hình không học thuộc "rover số k"
    strips = np.linspace(0, L, N + 1)
    p = np.stack([lawnmower_path(strips[order[i]], strips[order[i] + 1], L,
                                 cfg.lane_gap, cfg.speed, T, rng) for i in range(N)], axis=1)  # (T,N,2)

    # --- chọn lỗi
    label = np.zeros((T, N), np.int8)
    fault_type, faulty, t_u, rate_faulty = "none", np.array([], int), -1, 0.0
    if rng.random() < cfg.p_fault:
        types = list(cfg.fault_mix)
        fault_type = rng.choice(types, p=np.array([cfg.fault_mix[k] for k in types]))
        faulty = rng.choice(N, cfg.n_faulty, replace=False)
        t_u = int(rng.integers(cfg.t_fault[0], cfg.t_fault[1] + 1))
        rate_faulty = float(rng.uniform(*cfg.drift_rate))

    # --- vị trí tự khai: mọi rover trôi nhẹ; rover F-drift trôi mạnh từ t_u
    p_hat = p.copy()
    step_len = np.r_[np.zeros((1, N)), np.linalg.norm(np.diff(p, axis=0), axis=2)]  # (T,N)
    rate = rng.uniform(0, cfg.honest_drift_max, N)
    theta = rng.uniform(0, 2 * np.pi, N)
    off = np.zeros((N, 2))
    for t in range(T):
        theta += rng.normal(0, cfg.drift_turn_std, N)
        r_t = rate.copy()
        if fault_type == "drift" and t >= t_u:
            r_t[faulty] = rate_faulty
        off += (r_t * step_len[t])[:, None] * np.c_[np.cos(theta), np.sin(theta)]
        p_hat[t] = p[t] + off

    # --- khai giả
    spoof_target = None
    if fault_type in ("spoof", "spoof_one"):
        for k in faulty:
            # mục tiêu giả: điểm đối xứng qua tâm vùng (một vùng khác hẳn chỗ thật)
            spoof_target = np.array([L, L]) - p[t_u, k]
            if fault_type == "spoof":
                fake = p_hat[t_u, k].copy()
                for t in range(t_u, T):
                    d = spoof_target - fake
                    n = np.linalg.norm(d)
                    if n > cfg.speed:
                        fake = fake + d / n * cfg.speed
                    p_hat[t, k] = fake
            else:
                others = [j for j in range(N) if j != k]
                j_star = others[int(np.argmin(np.linalg.norm(p[t_u, others] - p[t_u, k], axis=1)))]
                phi0 = np.arctan2(*(p[t_u, k] - p_hat[t_u, j_star])[::-1])
                phi1 = np.arctan2(*(spoof_target - p_hat[t_u, j_star])[::-1])
                dphi = (phi1 - phi0 + np.pi) % (2 * np.pi) - np.pi
                for t in range(t_u, T):
                    a = min(1.0, (t - t_u) / 30.0)  # xoay dần trong 30 bước cho khó phát hiện
                    phi = phi0 + a * dphi
                    rad = np.linalg.norm(p[t, k] - p[t, j_star])  # khớp khoảng cách thật tới j*
                    p_hat[t, k] = p_hat[t, j_star] + rad * np.array([np.cos(phi), np.sin(phi)])
        meta_extra = {"spoof_receiver": int(j_star) if fault_type == "spoof_one" else -1}
    else:
        meta_extra = {"spoof_receiver": -1}

    if len(faulty):
        label[t_u:, faulty] = DRIFT if fault_type == "drift" else SPOOF

    # --- đo UWB (two-way ranging: một giá trị chung cho cả hai đầu của cặp)
    iu, ju = np.triu_indices(N, 1)
    d_true = np.linalg.norm(p[:, iu] - p[:, ju], axis=2)            # (T, P)
    bias = noise.pair_bias(rng, len(iu))                              # (P,)
    meas = d_true + bias[None] + noise.residuals(rng, d_true.shape)
    valid = (d_true <= cfg.comm_range) & (rng.random(d_true.shape) >= cfg.p_drop)
    meas = np.where(valid, meas, np.nan)
    r = np.full((T, N, N), np.nan, np.float32)
    r[:, iu, ju] = meas
    r[:, ju, iu] = meas

    meta = {"fault_type": fault_type, "faulty": faulty.tolist(), "t_u": t_u,
            "drift_rate": rate_faulty, "honest_rates": rate.tolist(), **meta_extra}
    return {"p_true": p.astype(np.float32), "p_hat": p_hat.astype(np.float32),
            "r": r, "label": label}, meta


def config_dict(cfg):
    return asdict(cfg)
