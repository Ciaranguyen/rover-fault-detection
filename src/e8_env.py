"""E8: môi trường phủ vùng đa rover có robot báo sai, để đo giá trị của bộ phát hiện ở cấp nhiệm vụ.

Bối cảnh (giống mô phỏng của E0–E7): 5 rover, vùng 24 m x 24 m, bước 1 s, 400 bước, tốc độ 0,3 m/s,
rover trung thực trôi định vị 0–0,5% quãng đường, 70% episode có 1 rover lỗi (drift / spoof / spoof_one,
cùng công thức với src/sim.py), đo UWB với nhiễu thật TIERS (src/sim.NoiseModel).

Nhiệm vụ: phủ hết vùng. Vùng chia lưới 16 x 16 ô (1,5 m = bề rộng làn quét), gom thành 8 khu (4 cột x 2 hàng,
mỗi khu 32 ô). Mỗi rover phát vị trí tự khai của mình; đội dựng một "bản đồ đã phủ" chung từ các vị trí tự khai
ĐƯỢC TIN. Rover lỗi làm bản đồ sai: khai giả -> đánh dấu đã phủ những ô chưa ai tới (lỗ hổng);
trôi -> đánh dấu lệch chỗ.

Ra quyết định (MARL): cứ 20 bước, mỗi rover chọn 1 trong 8 khu để làm tiếp. Bộ điều khiển cấp thấp cố định:
đi tới ô chưa phủ gần nhất (theo bản đồ chung + những ô chính mình đã khai) trong khu đã chọn.
Phần thưởng chung = 10 x phần tăng độ phủ THẬT (ô có rover thật sự đi qua) — chỉ dùng khi huấn luyện tập trung.

Nguồn tin cậy (trust source), quyết định rover nào bị coi là lỗi ở mỗi mốc quyết định:
  none     : tin tất cả
  oracle   : biết đúng rover lỗi từ lúc lỗi bắt đầu
  noisy    : oracle có trễ ngẫu nhiên, bỏ sót, báo nhầm ngẫu nhiên (chỉ dùng để HUẤN LUYỆN chính sách,
             để chính sách không phụ thuộc một bộ phát hiện cụ thể)
  callback : một hàm nhận dữ liệu đến hiện tại (p_hat, r) và trả cờ — dùng để cắm bộ phát hiện thật (e8_eval.py)
Khi một rover bị gắn cờ: bỏ qua mọi vị trí nó tự khai trong RETRO = 100 bước trước đó và trong suốt thời gian bị cờ.
"""
import numpy as np

from sim import NoiseModel, SimConfig

L, G, CELL = 24.0, 16, 1.5
R = 8                       # số khu
T, EPOCH = 400, 20          # bước mỗi episode, bước mỗi mốc quyết định
N_DEC = T // EPOCH
RETRO = 100                 # khi bị gắn cờ, bỏ các vị trí khai trong 100 bước trước (chốt trước khi chạy)
DET_START = 30              # bộ phát hiện chỉ có kết quả từ bước 29 (cửa sổ 30 bước, như E1–E7)
OBS_DIM = R + 2 + 1 + R + R + 1
GLOBAL_EXTRA = R + 10

_cx, _cy = np.meshgrid(np.arange(G), np.arange(G), indexing="xy")
CELL_REGION = ((_cy // 8) * 4 + (_cx // 4)).ravel()                       # (G*G,)
REG_CELLS = np.stack([np.where(CELL_REGION == r)[0] for r in range(R)])   # (R, 32)
CELL_CENTER = np.stack([(_cx.ravel() + 0.5) * CELL, (_cy.ravel() + 0.5) * CELL], -1)  # (G*G, 2)
REG_CENTER = CELL_CENTER[REG_CELLS].mean(1)                               # (R, 2)


def cell_of(xy):
    """xy (..., 2) -> chỉ số ô, -1 nếu ngoài vùng."""
    c = np.floor(xy / CELL).astype(int)
    ok = (c >= 0).all(-1) & (c < G).all(-1)
    return np.where(ok, c[..., 1] * G + c[..., 0], -1)


class CoverageEnv:
    def __init__(self, B, noise_file="data/noise_tiers.npz", n_rovers=5, p_fault=0.7, with_ranges=False,
                 fault_mix=None):
        self.B, self.N = B, n_rovers
        self.cfg = SimConfig(n_rovers=n_rovers, p_fault=p_fault)
        if fault_mix:
            self.cfg.fault_mix = fault_mix
        self.with_ranges = with_ranges
        self.noise = NoiseModel(noise_file, self.cfg) if with_ranges else None
        self.iu, self.ju = np.triu_indices(n_rovers, 1)

    # khởi tạo
    def reset(self, seeds):
        """seeds: list B số nguyên — mỗi episode một rng riêng, để mọi điều kiện đánh giá dùng CÙNG episode."""
        B, N, cfg = self.B, self.N, self.cfg
        self.rngs = [np.random.default_rng([777, int(s)]) for s in seeds]
        self.t = 0
        self.p = np.stack([r.uniform(1.0, L - 1.0, (N, 2)) for r in self.rngs])          # vị trí thật
        self.off = np.zeros((B, N, 2))
        self.theta = np.stack([r.uniform(0, 2 * np.pi, N) for r in self.rngs])
        self.rate = np.stack([r.uniform(0, cfg.honest_drift_max, N) for r in self.rngs])
        types = list(cfg.fault_mix)
        probs = np.array([cfg.fault_mix[k] for k in types])
        self.ftype = np.full(B, "none", dtype=object)
        self.faulty = np.full(B, -1)
        self.t_u = np.full(B, 10 ** 6)
        self.frate = np.zeros(B)
        for b, r in enumerate(self.rngs):
            if r.random() < cfg.p_fault:
                self.ftype[b] = types[r.choice(len(types), p=probs)]
                self.faulty[b] = r.integers(N)
                self.t_u[b] = r.integers(cfg.t_fault[0], cfg.t_fault[1] + 1)
                self.frate[b] = r.uniform(*cfg.drift_rate)
        self.spoof_target = np.zeros((B, 2))
        self.fake = np.zeros((B, 2))
        self.jstar = np.full(B, -1)
        self.phi0 = np.zeros(B)
        self.dphi = np.zeros(B)
        # lịch sử
        self.claims = np.zeros((B, T, N, 2), np.float32)
        self.claim_cell = np.full((B, T, N), -1)
        self.label = np.zeros((B, T, N), np.int8)
        self.untrusted = np.zeros((B, T, N), bool)
        self.flag = np.zeros((B, N), bool)
        self.true_cov = np.zeros((B, G * G), bool)
        self.belief = np.zeros((B, G * G), bool)          # bản đồ chung từ vị trí khai được tin
        self.own = np.zeros((B, N, G * G), bool)          # ô chính rover đã khai (nó tự tin)
        self.region = np.stack([r.integers(R, size=N) for r in self.rngs])
        self.target = np.full((B, N), -1)
        self.n_false_flags = np.zeros(B)
        self.n_flag_checks = np.zeros(B)
        if self.with_ranges:
            P = len(self.iu)
            self.r = np.full((B, T, N, N), np.nan, np.float32)
            self.bias = np.stack([self.noise.pair_bias(r, P) for r in self.rngs])                    # (B,P)
            self.resid = np.stack([self.noise.residuals(r, (T, P)) for r in self.rngs])             # (B,T,P)
            self.drop = np.stack([r.random((T, P)) < cfg.p_drop for r in self.rngs])                 # (B,T,P)
        self._noisy_setup()
        return self.obs()

    def _noisy_setup(self):
        """Tham số nguồn tin cậy 'noisy' cho huấn luyện: trễ, bỏ sót, tỉ lệ báo nhầm — mỗi episode khác nhau."""
        self.nz_delay = np.array([r.integers(0, 61) for r in self.rngs])
        self.nz_miss = np.array([r.random() < 0.1 for r in self.rngs])
        self.nz_pfa = np.array([r.uniform(0, 0.04) for r in self.rngs])

    # quan sát
    def nav_pos(self):
        return self.p + self.off             # vị trí rover tự tin (khung định vị của chính nó)

    def obs(self):
        B, N = self.B, self.N
        cov_reg = self.belief[:, REG_CELLS].mean(-1)                                   # (B,R)
        others = np.zeros((B, N, R))
        for k in range(N):
            m = ~self.flag                                                            # ai đang được tin
            m[:, k] = False
            onehot = np.eye(R)[self.region] * m[..., None]                             # (B,N,R)
            others[:, k] = onehot.sum(1) / (N - 1)
        o = np.concatenate([
            np.repeat(cov_reg[:, None], N, 1),
            np.clip(self.nav_pos() / L, 0, 1),
            np.full((B, N, 1), self.t / T),
            np.eye(R)[self.region],
            others,
            self.flag[..., None].astype(float),
        ], -1).astype(np.float32)
        return o                                                                      # (B,N,OBS_DIM)

    def global_extra(self):
        tc = self.true_cov[:, REG_CELLS].mean(-1)                                      # (B,R)
        pos = (self.p / L).reshape(self.B, -1)                                         # (B,2N)
        return np.concatenate([tc, pos], -1).astype(np.float32)                        # (B,R+2N)

    # nguồn tin cậy
    def flags_from(self, source, callback=None):
        B, N, t = self.B, self.N, self.t
        fl = np.zeros((B, N), bool)
        has = self.faulty >= 0
        if source == "oracle":
            fl[has, self.faulty[has]] = t >= self.t_u[has]
        elif source == "noisy":
            on = has & (t >= self.t_u + self.nz_delay) & ~self.nz_miss
            fl[on, self.faulty[on]] = True
            fa = np.stack([r.random(N) for r in self.rngs]) < self.nz_pfa[:, None]
            fl |= fa
        elif source == "callback" and t >= DET_START:
            fl = callback(self)
        return fl

    def apply_flags(self, fl):
        t = self.t
        is_f = (np.arange(self.N)[None] == self.faulty[:, None]) & (t >= self.t_u)[:, None]
        honest = ~is_f
        self.n_false_flags += (fl & honest).sum(1)
        self.n_flag_checks += honest.sum(1)
        self.flag = fl
        lo = max(0, t - RETRO)
        for b, k in zip(*np.where(fl)):
            self.untrusted[b, lo:t, k] = True
        self._rebuild_belief()

    def _rebuild_belief(self):
        t = self.t
        cc = self.claim_cell[:, :t]
        ok = (cc >= 0) & ~self.untrusted[:, :t]
        self.belief[:] = False
        bi = np.broadcast_to(np.arange(self.B)[:, None, None], cc.shape)
        self.belief[bi[ok], cc[ok]] = True

    # một bước 1 giây
    def _step1(self):
        B, N, t, cfg = self.B, self.N, self.t, self.cfg
        q = self.nav_pos()
        # chọn ô đích: ô chưa phủ gần nhất trong khu đã chọn, theo bản đồ chung + ô của chính mình
        known = self.belief[:, None, :] | self.own                                      # (B,N,G*G)
        cells = REG_CELLS[self.region]                                                  # (B,N,32)
        unc = ~np.take_along_axis(known, cells, -1)                                     # (B,N,32)
        d = np.linalg.norm(CELL_CENTER[cells] - q[:, :, None, :], axis=-1)
        d = np.where(unc, d, np.inf)
        j = d.argmin(-1)
        has = np.isfinite(d.min(-1))
        tgt = np.where(has[..., None], CELL_CENTER[np.take_along_axis(cells, j[..., None], -1)[..., 0]],
                       REG_CENTER[self.region])
        vec = tgt - q
        dist = np.linalg.norm(vec, axis=-1, keepdims=True)
        spd = cfg.speed * np.stack([r.uniform(0.8, 1.2, N) for r in self.rngs])[..., None]
        step = np.minimum(spd, dist)
        move = np.where(dist > 1e-9, vec / np.maximum(dist, 1e-9) * step, 0.0)
        p_new = np.clip(self.p + move, 0.01, L - 0.01)
        step_len = np.linalg.norm(p_new - self.p, axis=-1)
        self.p = p_new
        # trôi định vị (giống sim.py)
        self.theta += np.stack([r.normal(0, cfg.drift_turn_std, N) for r in self.rngs])
        r_t = self.rate.copy()
        dr = (self.ftype == "drift") & (t >= self.t_u)
        r_t[dr, self.faulty[dr]] = self.frate[dr]
        self.off += (r_t * step_len)[..., None] * np.stack([np.cos(self.theta), np.sin(self.theta)], -1)
        claim = self.p + self.off
        # khai giả (giống sim.py)
        for b in np.where(np.isin(self.ftype, ["spoof", "spoof_one"]) & (t >= self.t_u))[0]:
            k = self.faulty[b]
            if t == self.t_u[b]:
                self.spoof_target[b] = np.array([L, L]) - self.p[b, k]
                self.fake[b] = claim[b, k]
                if self.ftype[b] == "spoof_one":
                    oth = [i for i in range(N) if i != k]
                    js = oth[int(np.argmin(np.linalg.norm(self.p[b, oth] - self.p[b, k], axis=1)))]
                    self.jstar[b] = js
                    self.phi0[b] = np.arctan2(*(self.p[b, k] - claim[b, js])[::-1])
                    phi1 = np.arctan2(*(self.spoof_target[b] - claim[b, js])[::-1])
                    self.dphi[b] = (phi1 - self.phi0[b] + np.pi) % (2 * np.pi) - np.pi
            if self.ftype[b] == "spoof":
                dd = self.spoof_target[b] - self.fake[b]
                n = np.linalg.norm(dd)
                if n > cfg.speed:
                    self.fake[b] = self.fake[b] + dd / n * cfg.speed
                claim[b, k] = self.fake[b]
            else:
                js = self.jstar[b]
                a = min(1.0, (t - self.t_u[b]) / 30.0)
                phi = self.phi0[b] + a * self.dphi[b]
                rad = np.linalg.norm(self.p[b, k] - self.p[b, js])
                claim[b, k] = claim[b, js] + rad * np.array([np.cos(phi), np.sin(phi)])
        lab = np.zeros((B, N), np.int8)
        has = (self.faulty >= 0) & (t >= self.t_u)
        lab[has, self.faulty[has]] = np.where(self.ftype[has] == "drift", 1, 2)
        self.label[:, t] = lab
        self.claims[:, t] = claim
        cc = cell_of(claim)
        self.claim_cell[:, t] = cc
        # cập nhật bản đồ: ô thật, ô tự khai của mỗi rover, bản đồ chung (nếu rover không bị cờ)
        tc = cell_of(self.p)
        bi = np.repeat(np.arange(B), N)
        self.true_cov[bi, tc.ravel()] = True
        okc = cc >= 0
        bb, kk = np.where(okc)
        self.own[bb, kk, cc[okc]] = True
        trust = okc & ~self.flag
        if self.flag.any():
            self.untrusted[:, t] = self.flag
        bb, kk = np.where(trust)
        self.belief[bb, cc[trust]] = True
        if self.with_ranges:
            dtrue = np.linalg.norm(self.p[:, self.iu] - self.p[:, self.ju], axis=-1)        # (B,P)
            meas = dtrue + self.bias + self.resid[:, t]
            valid = (dtrue <= cfg.comm_range) & ~self.drop[:, t]
            meas = np.where(valid, meas, np.nan)
            self.r[:, t, self.iu, self.ju] = meas
            self.r[:, t, self.ju, self.iu] = meas
        self.t += 1

    def step(self, actions, source="none", callback=None):
        """actions (B,N) khu được chọn. Áp cờ tin cậy ở đầu mốc, chạy 20 bước. Trả obs, reward (B,), done."""
        self.region = actions
        self.apply_flags(self.flags_from(source, callback))
        before = self.true_cov.mean(-1)
        for _ in range(EPOCH):
            self._step1()
        rew = 10.0 * (self.true_cov.mean(-1) - before)
        return self.obs(), rew.astype(np.float32), self.t >= T

    # thước đo cuối episode
    def metrics(self):
        holes = self.belief & ~self.true_cov
        return {"coverage": self.true_cov.mean(-1), "holes": holes.mean(-1),
                "has_fault": self.faulty >= 0, "ftype": self.ftype.astype(str),
                "false_flag_rate": self.n_false_flags / np.maximum(self.n_flag_checks, 1)}

    def data_prefix(self):
        """Dữ liệu tới bước hiện tại cho bộ phát hiện (giống định dạng npz của make_dataset)."""
        t = self.t
        return {"p_hat": self.claims[:, :t], "r": self.r[:, :t], "label": self.label[:, :t]}


def heuristic_actions(env):
    """Chính sách tham chiếu không học: chọn khu có độ phủ (theo bản đồ chung) thấp nhất,
    phạt khoảng cách và số rover khác đang làm khu đó."""
    o = env.obs()
    cov = o[..., :R]
    q = env.nav_pos()
    dist = np.linalg.norm(REG_CENTER[None, None] - q[:, :, None, :], axis=-1) / L
    others = o[..., R + 3 + R: R + 3 + 2 * R]
    score = cov + 0.5 * dist + 1.0 * others
    return score.argmin(-1)
