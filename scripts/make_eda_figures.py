"""Week-2 EDA figures F1-F5 (static, print). Palette/marks follow the dataviz reference palette:
categorical slots blue #2a78d6 / orange #eb6834; sequential blue ramp; neutral grays for context;
hairline solid grids; 2px lines; text in ink tokens, never series color. Table views: reports/figures/data/*.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import numpy as np
import pandas as pd

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
S1, S2 = "#2a78d6", "#eb6834"
NEUTRAL_1, NEUTRAL_2 = "#f0efec", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
OUT, DATA = "reports/figures", "reports/figures/data"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42,
})
MONTHS = list("JFMAMJJASOND")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def phase_shading(ax):
    ax.axvspan(2.5, 5.5, color=NEUTRAL_1, zorder=0, lw=0)
    ax.axvspan(5.5, 9.5, color=NEUTRAL_2, zorder=0, lw=0)


# ---------- F1 study area ----------
adm1 = gpd.read_file("data/external/boundaries/bgd_admin1.geojson")
coords = pd.read_csv("data/processed/bmd_station_coords.csv")
xw = pd.read_csv("data/processed/crosswalk_spatial.csv")
gh_bd = xw[xw.track == "ghcn_bangladesh"]
temp = pd.read_csv("reports/temperate_reference_stations.csv")
world = gpd.read_file("zip://data/external/naturalearth/ne_110m_admin_0_countries.zip")

fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_axes([0.0, 0.0, 0.42, 1.0])
adm1.plot(ax=ax, color=NEUTRAL_1, edgecolor=AXIS, linewidth=0.5)
ax.scatter(gh_bd.lon, gh_bd.lat, s=95, color=S2, edgecolor=SURF, linewidth=1.5, zorder=3, label="GHCN-Daily station (n=10)")
ax.scatter(coords.lon, coords.lat, s=26, color=S1, edgecolor=SURF, linewidth=1.2, zorder=4, label="BMD station (n=35)")
for name, dx, dy in (("Dhaka", 0.12, 0.05), ("Sylhet", 0.12, 0.05), ("Teknaf", 0.12, -0.05), ("Rajshahi", -0.12, 0.10)):
    r = coords[coords.station == name].iloc[0]
    ax.annotate(name, (r.lon, r.lat), xytext=(r.lon + dx, r.lat + dy), color=INK2, fontsize=7.5, ha="left" if dx > 0 else "right")
ax.set_axis_off()
ax.set_title("a  Bangladesh observation stations", pad=4)
ax.legend(loc="lower left", fontsize=7.5, handletextpad=0.3, borderaxespad=0.2)

ax2 = fig.add_axes([0.46, 0.08, 0.54, 0.84])
world.plot(ax=ax2, color=NEUTRAL_1, edgecolor=AXIS, linewidth=0.3)
ax2.scatter(temp.lon, temp.lat, s=22, color=S1, edgecolor=SURF, linewidth=1.0, zorder=3, label=f"Temperate reference (n={len(temp)})")
ax2.scatter(gh_bd.lon, gh_bd.lat, s=22, color=S2, edgecolor=SURF, linewidth=1.0, zorder=3, label="Bangladesh GHCN (n=10)")
ax2.axhspan(35, 60, color=GRID, alpha=0.35, zorder=0, lw=0)
ax2.set_xlim(-15, 160); ax2.set_ylim(5, 72)
ax2.set_xticks([]); ax2.set_yticks([35, 60]); ax2.set_yticklabels(["35°N", "60°N"])
for s in ("left", "bottom"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(length=0)
ax2.set_title("b  GHCN-Daily tracks: same WMO network, two climates", pad=4)
ax2.legend(loc="lower left", fontsize=7.5, handletextpad=0.3)
save(fig, "F1_study_area")

# ---------- F2 seasonal cycle (small multiples, one axis each) ----------
c = pd.read_csv(f"{DATA}/f2_seasonal_cycle.csv", index_col=0)
x = np.arange(1, 13)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 4.2), sharex=True, gridspec_kw={"hspace": 0.28})
for ax in (a1, a2):
    phase_shading(ax)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
a1.bar(x, c.rain_mm_median, width=0.55, color=S1, zorder=2)
a1.vlines(x, c.rain_mm_p25, c.rain_mm_p75, color=INK2, linewidth=1.0, zorder=3)
a1.set_ylabel("Monthly rainfall (mm)")
a1.set_title("a  Rainfall: station median and IQR")
peak = int(c.rain_mm_median.idxmax())
a1.annotate(f"{c.rain_mm_median.max():.0f} mm", (peak, c.rain_mm_p75.loc[peak]), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", color=INK2, fontsize=7)
a1.set_ylim(0, c.rain_mm_p75.max() * 1.12)
a2.fill_between(x, c.temp_c_p25, c.temp_c_p75, color=S1, alpha=0.10, lw=0, zorder=2)
a2.plot(x, c.temp_c_median, color=S1, linewidth=2, solid_capstyle="round", zorder=3)
a2.set_ylabel("Mean temperature (°C)")
a2.set_title("b  Temperature: station median and IQR")
a2.text(4, 18.0, "pre-monsoon", ha="center", va="bottom", color=MUTED, fontsize=7)
a2.text(7.5, 18.0, "monsoon", ha="center", va="bottom", color=MUTED, fontsize=7)
a2.set_xticks(x); a2.set_xticklabels(MONTHS)
fig.text(0.01, -0.01, "BMD, 35 stations, training period to 2010, suspected-imputed values excluded.", color=MUTED, fontsize=6.5)
save(fig, "F2_seasonal_cycle")

# ---------- F3 coverage heatmaps ----------
cov = pd.read_csv(f"{DATA}/f3_coverage_station_year.csv")
panels = [("BMD", "a  BMD national network (clean daily rainfall)"), ("GHCN bangladesh", "b  GHCN-Daily, Bangladesh"), ("GHCN temperate", "c  GHCN-Daily, temperate reference")]
heights = [cov[cov.track == t].station_id.nunique() for t, _ in panels]
fig, axes = plt.subplots(3, 1, figsize=(7.2, 11.5), gridspec_kw={"height_ratios": heights, "hspace": 0.22})
names = pd.read_csv("reports/temperate_reference_stations.csv").set_index("id").name.str.title().to_dict()
from bwb.data.ghcn import BG_TO_BMD
names.update({k: f"{v} (GHCN)" for k, v in BG_TO_BMD.items()})
years = np.arange(1961, 2026)
for ax, (track, title) in zip(axes, panels):
    d = cov[cov.track == track].pivot(index="station_id", columns="year", values="pct").reindex(columns=years)
    first = d.gt(0).idxmax(axis=1)
    d = d.apply(lambda row: row.where(row.index >= first[row.name]), axis=1)
    d = d.loc[first.sort_values().index]
    d.index = [names.get(i, i) for i in d.index]
    im = ax.imshow(d.to_numpy(), aspect="auto", cmap=SEQ, vmin=0, vmax=100, interpolation="nearest")
    ax.set_facecolor(NEUTRAL_1)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=6)
    ticks = [i for i, y in enumerate(years) if y % 10 == 0]
    ax.set_xticks(ticks); ax.set_xticklabels([years[i] for i in ticks])
    ax.axvspan(years.tolist().index(2016) - 0.5, years.tolist().index(2023) + 0.5, fill=False, edgecolor=INK2, linewidth=0.8)
    ax.set_title(title)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.015, pad=0.035, aspect=45)
cb.set_label("Days with a usable rainfall observation (%). Blank = no record. Outlined: 2016-2023 test period.", color=INK2)
cb.outline.set_visible(False)
save(fig, "F3_coverage")

# ---------- F4 rainfall distribution ----------
h = pd.read_csv(f"{DATA}/f4_wetday_histogram.csv")
sm = pd.read_csv(f"{DATA}/f4_rain_summary.csv").set_index("stat").value
fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.grid(axis="y"); ax.set_axisbelow(True)
ax.bar(h.bin_lo_mm, h.days, width=(h.bin_hi_mm - h.bin_lo_mm) * 0.85, align="edge", color=S1, zorder=2)
ax.set_ylim(1, h.days.max() * 8)
ax.set_xscale("log"); ax.set_yscale("log")
for q, lab in (("wet_p95_mm", "95th"), ("wet_p99_mm", "99th")):
    ax.axvline(sm[q], color=INK2, linewidth=0.8, zorder=3)
    left = lab == "95th"
    ax.text(sm[q] * (0.92 if left else 1.08), 2.5e4, f"{lab} pct\n{sm[q]:.0f} mm", color=INK2, fontsize=7, va="center", ha="right" if left else "left")
ax.set_xlabel("Daily rainfall on wet days (mm, log scale)")
ax.set_ylabel("Station-days (log scale)")
ax.set_title(f"Zero-inflated, heavy-tailed daily rainfall\n{sm['dry_day_share(<0.1mm)'] * 100:.1f}% of all days are dry (not shown)", fontsize=9)
save(fig, "F4_rainfall_distribution")

# ---------- F5 teleconnections ----------
t = pd.read_csv(f"{DATA}/f5_teleconnection_lag_corr.csv")
sig = t.abs_r_sig95.iloc[0]
fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.axhspan(-sig, sig, color=NEUTRAL_1, lw=0, zorder=0)
ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
for name, col in (("ONI", S1), ("DMI", S2)):
    d = t[t["index"] == name].sort_values("lead_months")
    ax.plot(d.lead_months, d.r, color=col, linewidth=2, zorder=3, label=f"{name} ({'ENSO' if name == 'ONI' else 'IOD'})")
    ax.scatter(d.lead_months, d.r, s=16, color=col, edgecolor=SURF, linewidth=1.0, zorder=4)
    last = d.iloc[-1]
    ax.text(last.lead_months + 0.3, last.r, name, color=INK2, fontsize=7, va="center")
lab = t[t["index"] == "ONI"].sort_values("lead_months")
ax.set_xticks(lab.lead_months[::2]); ax.set_xticklabels(lab.index_month[::2])
ax.set_xlim(-0.5, 13.3); ax.set_ylim(-0.45, 0.45)
ax.set_xlabel("Index month (Aug back to previous Aug)")
ax.set_ylabel("Correlation with JJAS rainfall anomaly")
ax.set_title("Weak ENSO/IOD links to Bangladesh monsoon rainfall")
ax.text(12.9, sig + 0.02, "p<0.05 threshold", ha="right", color=MUTED, fontsize=6.5)
ax.legend(loc="lower left", fontsize=7)
fig.text(0.01, -0.09, f"National mean of 35 BMD stations, 1961-2010 (n={int(t.n_years.iloc[0])} years). Shaded: not significant.", color=MUTED, fontsize=6.5)
save(fig, "F5_teleconnections")
print("figures written:", ", ".join(f"F{i}" for i in range(1, 6)))
