import re
import sys
import pandas as pd

# ====== CẤU HÌNH ======
PATH = r"C:\Users\ThisPC\Desktop\bang gia tri bien thie dien tro theo 10 chu ky.xlsx"
OUT = r"C:\Users\ThisPC\Desktop\gel_clean.csv"
L0 = 24.0             # chiều dài ban đầu (mm)
MM_PER_STEP = 0.5     # mỗi dòng dịch 0.5 mm
DEG_PER_STEP = 4.0    # mỗi dòng = 4 độ  (=> 8 độ / mm)
EXPECTED_ROWS = 25    # mỗi chiều 25 dòng: 0 -> 12 mm, bước 0.5
R_RANGE = (5, 20)     # chỉ dùng để CẢNH BÁO, không xoá dữ liệu
# =======================

# Vị trí cột (A=0, B=1, C=2, D=3, ..., H=7, I=8, J=9, K=10)
#   disp = cột dịch chuyển (mm) | L = chiều dài (mm) | R = điện trở (kΩ)
#   f* = chiều thuận, b* = chiều nghịch
# Chu kỳ 2-10: thuận A,B,D | nghịch H (dịch chuyển), I (R)  -> không có cột chiều dài
# Chu kỳ 1   : thuận A,B,D | nghịch H (dịch chuyển), I (chiều dài), J (strain), K (R)
LAYOUT_DEFAULT = dict(f_disp=0, f_L=1, f_R=3, b_disp=7, b_L=None, b_R=8)
LAYOUT_C1 = dict(f_disp=0, f_L=1, f_R=3, b_disp=7, b_L=8, b_R=10)
LAYOUTS = {1: LAYOUT_C1}    # chu kỳ nào không có trong đây dùng LAYOUT_DEFAULT


def colname(i):
    s = ""; i += 1
    while i:
        i, m = divmod(i - 1, 26)
        s = chr(65 + m) + s
    return s


def to_num(x):
    """Đổi sang số, chấp nhận dấu phẩy thập phân."""
    if isinstance(x, str):
        x = x.strip().replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


def longest_run(col):
    """(start, end) của đoạn ô có số liên tục dài nhất (end không gồm)."""
    best = (0, 0); start = None
    vals = col.values
    for i, v in enumerate(vals):
        if pd.notna(v):
            if start is None:
                start = i
        elif start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(vals) - start > best[1] - best[0]:
        best = (start, len(vals))
    return best


def read_block(num, c_disp, c_L, c_R, label, warns):
    """Đọc 1 chiều (thuận hoặc nghịch). Vùng dòng lấy theo cột dịch chuyển."""
    for c in (c_disp, c_L, c_R):
        if c is not None and c >= num.shape[1]:
            raise ValueError(f"{label}: sheet không có cột {colname(c)}")
    s, e = longest_run(num.iloc[:, c_disp])
    if e - s == 0:
        raise ValueError(f"{label}: cột {colname(c_disp)} (dịch chuyển) không có số")
    blk = pd.DataFrame({
        "disp": num.iloc[s:e, c_disp].values,
        "R_kohm": num.iloc[s:e, c_R].values,
    })
    derived = blk["disp"] + L0
    if c_L is not None:
        blk["length_mm"] = num.iloc[s:e, c_L].values
        if (blk["length_mm"] - derived).abs().max() > 0.01:
            warns.append(f"{label}: cột chiều dài {colname(c_L)} lệch so với {L0:g}+dịch chuyển")
    else:
        blk["length_mm"] = derived.values          # chiều dài = 24 + dịch chuyển

    rows = f"{s + 1}-{e}"
    if len(blk) != EXPECTED_ROWS:
        warns.append(f"{label}: {len(blk)} dòng (dòng Excel {rows}), mong đợi {EXPECTED_ROWS}")
    if blk.isna().any().any():
        warns.append(f"{label}: có ô trống/chữ trong vùng số liệu (dòng Excel {rows})")
    bad = blk[(blk.R_kohm < R_RANGE[0]) | (blk.R_kohm > R_RANGE[1])]
    if len(bad):
        warns.append(f"{label}: {len(bad)} giá trị R ngoài {R_RANGE} -> {bad.R_kohm.round(2).tolist()[:5]}")
    return blk


def build_cycle(f, b, cid):
    f = f.copy(); b = b.copy()
    f["direction"] = 1
    b["direction"] = -1
    # điểm đầu nghịch trùng điểm cuối thuận (đỉnh 12 mm) -> bỏ
    if (len(b) and abs(b.disp.iloc[0] - f.disp.iloc[-1]) < 1e-9
            and abs(b.R_kohm.iloc[0] - f.R_kohm.iloc[-1]) < 1e-9):
        b = b.iloc[1:]
    out = pd.concat([f, b], ignore_index=True)
    out["angle_deg"] = out["disp"] / MM_PER_STEP * DEG_PER_STEP
    out.insert(0, "cycle_id", cid)
    out.insert(1, "step", range(len(out)))
    return out[["cycle_id", "step", "direction", "angle_deg", "length_mm", "R_kohm"]]


def sheet_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 10**6


if __name__ == "__main__":
    if len(sys.argv) > 1: PATH = sys.argv[1]
    if len(sys.argv) > 2: OUT = sys.argv[2]

    xl = pd.read_excel(PATH, sheet_name=None, header=None, dtype=object)
    names = sorted(xl.keys(), key=sheet_number)
    print("Các sheet tìm thấy:", names)

    parts = []
    for sheet in names:
        n = sheet_number(sheet)
        cid = f"c{n:02d}"
        lay = LAYOUTS.get(n, LAYOUT_DEFAULT)
        num = xl[sheet].apply(lambda col: col.map(to_num))
        warns = []
        try:
            f = read_block(num, lay["f_disp"], lay["f_L"], lay["f_R"], "thuận", warns)
            b = read_block(num, lay["b_disp"], lay["b_L"], lay["b_R"], "nghịch", warns)
        except ValueError as err:
            print(f"[BỎ QUA] '{sheet}': {err}")
            continue
        parts.append(build_cycle(f, b, cid))
        kind = "bố cục chu kỳ 1" if n in LAYOUTS else "bố cục chu kỳ 2-10"
        print(f"'{sheet}' -> {cid} ({kind}): thuận {len(f)} dòng | nghịch {len(b)} dòng")
        for w in warns:
            print(f"    ⚠ {w}")

    if not parts:
        raise SystemExit("Không đọc được sheet nào.")

    data = pd.concat(parts, ignore_index=True)
    data.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"\nĐã ghi {OUT}: {len(data)} dòng, {data.cycle_id.nunique()} chu kỳ")
    for cid, g in data.groupby("cycle_id"):
        flags = []
        f_ = g[g.direction == 1]; b_ = g[g.direction == -1]
        if f_.length_mm.iloc[-1] < f_.length_mm.iloc[0]: flags.append("thuận L đang giảm")
        if b_.length_mm.iloc[-1] > b_.length_mm.iloc[0]: flags.append("nghịch L đang tăng")
        if g.angle_deg.iloc[0] != g.angle_deg.iloc[-1]: flags.append("góc đầu≠cuối")
        print(f"  {cid}: {len(g)} điểm, góc {g.angle_deg.min():.0f}-{g.angle_deg.max():.0f}°, "
              f"R {g.R_kohm.min():.2f}-{g.R_kohm.max():.2f} kΩ, "
              f"L {g.length_mm.min():.1f}-{g.length_mm.max():.1f} mm"
              + (f"  ⚠ {', '.join(flags)}" if flags else "  ✓"))