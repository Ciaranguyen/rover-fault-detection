"""Trích mô hình nhiễu đo khoảng cách UWB từ bộ dữ liệu TIERS (MIT License).

Nguồn: https://github.com/TIERS/uwb-relative-localization-dataset
Bài báo: Torrico Morón et al., "Benchmarking UWB-Based Infrastructure-Free Positioning
and Multi-Robot Relative Localization: Dataset and Characterization", arXiv:2305.08532.

Với mỗi phép đo UWB giữa robot a và b, so với khoảng cách thật từ OptiTrack:
    sai số = r_uwb - ||p_a - p_b||
Tách thành: độ lệch riêng của từng cặp (trung vị sai số của cặp) + phần dư ngẫu nhiên.
Đầu ra: data/noise_tiers.npz gồm
    bias      : độ lệch của từng cặp (m)
    residual  : phần dư sau khi trừ độ lệch cặp (m)
    dist      : khoảng cách thật tương ứng với từng phần dư (m)
    source    : case lấy dữ liệu (3 = robot đứng yên, 4 = robot di chuyển)
    series, series_offsets, series_source :
                chuỗi phần dư của từng cặp, lấy mẫu lại 1 Hz, giữ nguyên thứ tự thời gian
                (nhiễu UWB thật tương quan theo thời gian: trễ 1 s tương quan khoảng 0,4–0,5)

Chạy:
    python src/noise_tiers.py --tiers <đường dẫn repo TIERS> --out data/noise_tiers.npz
"""
import argparse
import re
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def read_bag(bag_dir):
    """Đọc một rosbag2: trả về dict khoảng cách theo cặp và dict vị trí theo robot."""
    ranges, poses = {}, {}
    with AnyReader([Path(bag_dir)], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections
                 if c.msgtype in ("sensor_msgs/msg/Range", "geometry_msgs/msg/PoseStamped")]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = t_ns * 1e-9  # dùng mốc thời gian ghi bag cho cả hai loại tin nhắn
            if conn.msgtype == "sensor_msgs/msg/Range":
                a, b = map(int, re.findall(r"n_(\d+)", conn.topic))
                ranges.setdefault((a, b), []).append((t, msg.range))
            else:
                name = conn.topic.split("/")[2]  # /vrpn_client_node/turtlebotK/pose
                p = msg.pose.position
                poses.setdefault(name, []).append((t, p.x, p.y, p.z))
    return ({k: np.asarray(v) for k, v in ranges.items()},
            {k: np.asarray(v) for k, v in poses.items()})


def interp_pose(track, t):
    return np.stack([np.interp(t, track[:, 0], track[:, k]) for k in (1, 2, 3)], axis=1)


def errors_from_bag(bag_dir, min_samples=20):
    """Sai số theo từng cặp. Giả định nút UWB n_k gắn trên turtlebotk
    (đã kiểm tra ở case 4: cách ghép này cho sai số nhỏ nhất trong mọi hoán vị)."""
    ranges, poses = read_bag(bag_dir)
    out = []
    for (a, b), r in ranges.items():
        ta, tb = f"turtlebot{a}", f"turtlebot{b}"
        if ta not in poses or tb not in poses:
            continue
        t = r[:, 0]
        # chỉ giữ phép đo nằm trong khoảng thời gian có vị trí của cả hai robot
        ok = ((t >= max(poses[ta][0, 0], poses[tb][0, 0])) &
              (t <= min(poses[ta][-1, 0], poses[tb][-1, 0])))
        t, rr = t[ok], r[ok, 1]
        if len(t) < min_samples:
            continue
        g = np.linalg.norm(interp_pose(poses[ta], t) - interp_pose(poses[tb], t), axis=1)
        e = rr - g
        keep = np.isfinite(e)
        out.append(((a, b), e[keep], g[keep], t[keep]))
    return out


def resample_1hz(t, x):
    """Lấy mẫu chuỗi về 1 Hz bằng giá trị đo gần nhất (giữ tương quan thời gian thật)."""
    grid = np.arange(t[0], t[-1], 1.0)
    j = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    return x[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", required=True, help="thư mục repo TIERS đã clone")
    ap.add_argument("--out", default="data/noise_tiers.npz")
    ap.add_argument("--cases", default="3,4", help="các case dùng, ví dụ 3,4")
    args = ap.parse_args()

    bias, resid, dist, src = [], [], [], []
    series, series_src = [], []
    for case in [int(c) for c in args.cases.split(",")]:
        for bag in sorted((Path(args.tiers) / f"case_{case}" / "ros2").iterdir()):
            if not (bag / "metadata.yaml").exists():
                continue
            for _, e, g, t in errors_from_bag(bag):
                b = np.median(e)
                bias.append(b)
                resid.append(e - b)
                dist.append(g)
                src.append(np.full(len(e), case))
                s1 = resample_1hz(t, e - b)
                if len(s1) >= 30:
                    series.append(s1.astype(np.float32))
                    series_src.append(case)
            print(f"case {case} {bag.name}: xong")

    bias = np.asarray(bias)
    resid, dist, src = map(np.concatenate, (resid, dist, src))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(x) for x in series])
    np.savez_compressed(args.out, bias=bias, residual=resid, dist=dist, source=src,
                        series=np.concatenate(series), series_offsets=offsets,
                        series_source=np.asarray(series_src))

    robust = 1.4826 * np.median(np.abs(resid))
    print(f"\nSố cặp: {len(bias)}  số phép đo: {len(resid)}")
    print(f"Độ lệch cặp: trung vị {np.median(bias):.3f} m, IQR {np.percentile(bias, 25):.3f}..{np.percentile(bias, 75):.3f} m")
    print(f"Phần dư: std {resid.std():.3f} m, sigma bền vững {robust:.3f} m, "
          f"tỉ lệ |dư| > 3 sigma bền vững {(np.abs(resid) > 3 * robust).mean():.3f}")
    for case in np.unique(src):
        r = resid[src == case]
        print(f"  case {case}: {len(r)} phép đo, std {r.std():.3f}, sigma bền vững {1.4826*np.median(np.abs(r)):.3f}")
    lens = np.diff(offsets)
    print(f"Chuỗi 1 Hz: {len(series)} chuỗi, dài trung vị {int(np.median(lens))} s")
    s1 = np.concatenate([x for x, c in zip(series, series_src) if c == 4])
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
