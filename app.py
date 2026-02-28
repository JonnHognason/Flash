import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Basera á: Fair & Kormos, J. Chromatogr. A 1211 (2008) 49–54

st.set_page_config(page_title="Flash frá TLC forsøgn", layout="wide")
st.title("Flash column forsøgn frá TLC dátum")
st.caption("Predict VR*, band rúmd, fraction windows, og vís Gaussian chromatogram.")


# Líkningarnar sum eru nýttar í artiklinum
def rf(distance_spot: float, distance_front: float) -> float:
    return distance_spot / distance_front

def k_from_rf(rf_val: float) -> float:
    return (1.0 - rf_val) / rf_val     # Eq (1): k = (1-Rf)/Rf

def silica_mass(sample_mass_g: float, difficulty: str, min_rf: float) -> float:
    # Eqs (4) & (5)
    if difficulty == "easy":
        return 59.8 * sample_mass_g
    if difficulty == "hard":
        return 151.2 * sample_mass_g + 0.5
    # auto
    return 59.8 * sample_mass_g if min_rf >= 0.20 else (151.2 * sample_mass_g + 0.5)

def void_volume(silica_g: float) -> float:
    # Eq (6)
    return 1.8 * silica_g + 0.3

def correction_coefficient(column_pack: str) -> float:
    # miðal C: 0.64 manual; 0.66 commercial
    return 0.64 if column_pack == "manual" else 0.66

def vr_corrected(vv: float, rf_val: float, C: float) -> float:
    # Eq (3): V_R* = (Vv/Rf) * C
    return (vv / rf_val) * C

def efficiency_n(xa: float, difficulty: str) -> float:
    # Eqs (8) & (9). Viðmælt X_A input ~0.1–1.0.
    x_clamped = min(max(xa, 0.1), 1.0)
    if difficulty == "hard":
        return 51.70 * (x_clamped ** (-0.44))
    return 33.64 * (x_clamped ** (-0.44))  # easy

def band_volume(vr: float, n_eff: float) -> float:
    # Eq (7): Vb = 4*VR*/sqrt(N)
    return 4.0 * vr / math.sqrt(n_eff)

def resolution(vr_a: float, vb_a: float, vr_b: float, vb_b: float) -> float:
    # Eq (10)
    return 2.0 * (vr_b - vr_a) / (vb_b + vb_a)

def rs_label(rs: float) -> str:
    if rs < 0.8:
        return "Vánaligt"
    if rs < 1.5:
        return "Miðal"
    return "Góð"

def predicted_fraction_range(vr: float, vb: float, frac_size_ml: float):
    mid = max(1, int(round(vr / frac_size_ml)))
    width = max(1, int(round(vb / frac_size_ml)) + 1)
    half = width // 2
    start = max(1, mid - half)
    end = max(start, mid + half)
    return mid, start, end, width


# UI
with st.sidebar:
    st.header("Inntak")

    st.subheader("TLC Mát")
    distance_front = st.number_input(
        "Loysingarevni front mát (mm)",
        min_value=0.001, value=35.0, step=0.5
    )

    n_analytes = st.selectbox("Tal av analytes", options=[2, 3, 4, 5], index=1)

    nøvn, longd, xas = [], [], []
    for i in range(n_analytes):
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            nøvn.append(st.text_input(f"Analyte {i+1} navn", value=f"Analyte_{i+1}"))
        with col2:
            longd.append(st.number_input(
                f"Longdar mát av spot {i+1}", min_value=0.0, value=10.0 if i == 0 else 7.0,
                step=0.5, key=f"d{i}"
            ))
        with col3:
            xas.append(st.number_input(
                f"Massa ratio {i+1}", min_value=0.0, value=1.0/n_analytes,
                step=0.05, key=f"xa{i}"
            ))

    st.subheader("Sample og column informatión")
    sample_mass = st.number_input(
        "Samla crude nøgd (g)",
        min_value=0.0001, value=0.16, step=0.01, format="%.4f"
    )

    difficulty = st.selectbox(
        "Separatión",
        options=["auto", "easy", "hard"], index=0
    )

    column_pack = st.selectbox(
        "Column pakking",
        options=["manual", "commercial"], index=0
    )

    st.subheader("Overides")
    override_silica = st.checkbox("Override silica massi (g)")
    silica_override_val = None
    if override_silica:
        silica_override_val = st.number_input("Override silica massi (g)", min_value=0.01, value=10.0, step=0.5)

    override_fraction = st.checkbox("Override fractión volumin (mL)")
    fraction_override_val = None
    if override_fraction:
        fraction_override_val = st.number_input("Fractión fractión volumin (mL)", min_value=0.1, value=6.0, step=0.5)

    override_C = st.checkbox("Override correction koeffisientur C")
    C_override_val = None
    if override_C:
        C_override_val = st.number_input("C override", min_value=0.1, max_value=1.0, value=0.64, step=0.01)

    st.subheader("Chromatogram display")
    extra_ml = st.number_input(
        "Eyka mL aftaná síðstu fraktón á grafinum",
        min_value=0.0, value=5.0, step=1.0
    )
    points_per_ml = st.slider("Plot resolution", min_value=10, max_value=200, value=60, step=10)




# Útrokningar
df = pd.DataFrame({"Analyte": nøvn, "Distance": longd, "X_A": xas})
df["Rf"] = df["Distance"].apply(lambda d: rf(d, distance_front))
df["k"] = df["Rf"].apply(k_from_rf)

min_rf = float(df["Rf"].min())

difficulty_resolved = ("easy" if min_rf >= 0.20 else "hard") if difficulty == "auto" else difficulty
C = float(C_override_val) if C_override_val is not None else correction_coefficient(column_pack)

silica_g = float(silica_override_val) if silica_override_val is not None else silica_mass(sample_mass, difficulty_resolved, min_rf)
vv = void_volume(silica_g)

fraction_size = float(fraction_override_val) if fraction_override_val is not None else (vv / 3.0)

df["V_R* (mL)"] = df["Rf"].apply(lambda r: vr_corrected(vv, r, C))
df["N (efficiency)"] = df["X_A"].apply(lambda xa: efficiency_n(float(xa), difficulty_resolved))
df["V_b (mL)"] = df.apply(lambda row: band_volume(float(row["V_R* (mL)"]), float(row["N (efficiency)"])), axis=1)

pred = df.apply(lambda row: predicted_fraction_range(float(row["V_R* (mL)"]), float(row["V_b (mL)"]), fraction_size), axis=1)
df["Frac mid"] = [p[0] for p in pred]
df["Frac start"] = [p[1] for p in pred]
df["Frac end"] = [p[2] for p in pred]
df["Band width (fractions)"] = [p[3] for p in pred]

df_sorted = df.sort_values("V_R* (mL)").reset_index(drop=True)

# Rs tabel
rs_rows = []
for i in range(len(df_sorted) - 1):
    a = df_sorted.loc[i]
    b = df_sorted.loc[i + 1]
    rs = resolution(float(a["V_R* (mL)"]), float(a["V_b (mL)"]), float(b["V_R* (mL)"]), float(b["V_b (mL)"]))
    rs_rows.append({
        "Pær": f'{a["Analyte"]} vs {b["Analyte"]}',
        "Rs": rs,
        "Kvalitetur": rs_label(rs),
        "Glopp (fractiónir)": int(b["Frac start"] - a["Frac end"] - 1),
    })
rs_df = pd.DataFrame(rs_rows)


# Display
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Viðmældir parametrar")
    st.write(
        {
            "Difficulty": difficulty_resolved,
            "Correction koeffisientur C": round(C, 2),
            "Silica massi (g)": round(silica_g, 2),
            "Void volumin Vv (mL)": round(vv, 2),
            "Ideal fractión volumin (mL)": round(vv / 3.0, 2),
            "Brúktur fractión volumin (mL)": round(fraction_size, 2),
        }
    )

    if min_rf < 0.08:
        st.error("min Rf < 0.08 — Greinin viðmælir ikki at nýta eina loypivesku við hesum parametrinum (ógvuligani breiðir toppar).")
    elif min_rf < 0.10:
        st.warning("min Rf < 0.10 — Topparnir vera breiðir.")

    if not math.isclose(sum(xas), 1.0, rel_tol=0.15, abs_tol=0.15):
        st.warning(f"Massa ratio summurin er {sum(xas):.2f} (gerinin viðmælir ≈ 1.0).")

    st.subheader("Analyte forsøgn (sortera eftur elution)")
    show_cols = [
        "Analyte", "Distance", "Rf", "k", "X_A",
        "V_R* (mL)", "N (efficiency)", "V_b (mL)",
        "Frac start", "Frac end", "Frac mid", "Band width (fractions)"
    ]
    st.dataframe(
        df_sorted[show_cols].style.format({
            "Rf": "{:.3f}",
            "k": "{:.2f}",
            "X_a": "{:.2f}",
            "V_R* (mL)": "{:.2f}",
            "N (efficiency)": "{:.1f}",
            "V_b (mL)": "{:.2f}",
        }),
        use_container_width=True
    )

with right:
    st.subheader("Granna toppur resolution (Rs)")
    if rs_df.empty:
        st.info("Tað skullu minst tveir toppar til at rokna Rs.")
    else:
        st.dataframe(rs_df.style.format({"Rs": "{:.2f}"}), use_container_width=True)

    st.subheader("Gaussian chromatogram forsøgn")

    # x-axis max: koyr nakrir fleiri ml á grafin
    max_end_frac = int(df_sorted["Frac end"].max())
    x_max = max_end_frac * fraction_size + float(extra_ml)

    # x axis
    n_points = max(500, int(x_max * points_per_ml))
    x = np.linspace(0.0, x_max, n_points)

    # Gaussian toppar
    # Vb er breiddin ið er rokna frá VR* og N. Baseline breiddin ~ 4σ  => σ ≈ Vb/4.
    total = np.zeros_like(x)
    peaks = []

    for _, row in df_sorted.iterrows():
        mu = float(row["V_R* (mL)"])
        vb = float(row["V_b (mL)"])
        sigma = max(vb / 4.0, 1e-6)  # umganga at dividera við 0
        amp = float(row["X_A"])      # skalera toppa hædd við massa ratio

        y = amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        peaks.append((row["Analyte"], y, mu, sigma))
        total += y

    # Normalisera 
    maxy = float(total.max()) if float(total.max()) > 0 else 1.0
    total_n = total / maxy

    fig, ax = plt.subplots()
    ax.plot(x, total_n, label="_nolegend_")

    # Plotta
    for name, y, mu, sigma in peaks:
        ax.plot(x, y / maxy, label=str(name), alpha=0.9)

        row = df_sorted[df_sorted["Analyte"] == name].iloc[0]
        start_v = (int(row["Frac start"]) - 1) * fraction_size
        end_v = int(row["Frac end"]) * fraction_size

        ax.axvline(mu, linestyle="--", linewidth=1)
        ax.axvline(start_v, linestyle=":", linewidth=1)
        ax.axvline(end_v, linestyle=":", linewidth=1)

    ax.set_xlabel("Elution nøgd (mL)")
    ax.set_ylabel("Relativt signal (normalized)")
    ax.set_xlim(0, x_max)
    ax.legend(fontsize=8, loc="upper right")
    st.pyplot(fig)

st.caption("Lýkningarnir og Gaussian fordeilingin eru frá greimimi: Basera á: Fair & Kormos, J. Chromatogr. A 1211 (2008) 49–54.")
