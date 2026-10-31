"""Membuat dataset sintetis untuk sistem penilaian kelayakan kredit."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1200

usia = rng.integers(21, 60, N)
penghasilan = np.round(np.clip(rng.lognormal(2.2, 0.55, N), 3, 60), 1)   # juta rupiah/bulan
tanggungan = rng.integers(0, 6, N)
lama_kerja = np.clip(usia - 21 - rng.integers(0, 8, N), 0, 35)
riwayat = rng.choice(["Baik", "Cukup", "Buruk"], N, p=[0.55, 0.30, 0.15])
pinjaman = np.round(np.clip(penghasilan * rng.uniform(1, 12, N), 2, 300), 1)  # juta rupiah
tenor = rng.choice([6, 12, 24, 36, 48], N)
rasio_cicilan = np.round((pinjaman / tenor) / penghasilan, 3)  # cicilan per bulan / penghasilan

skor_riwayat = pd.Series(riwayat).map({"Baik": 1.2, "Cukup": 0.0, "Buruk": -1.8}).to_numpy()
z = (0.5 + skor_riwayat
     - 5.0 * rasio_cicilan
     + 0.08 * lama_kerja
     + 0.04 * np.log(penghasilan) * 10
     - 0.25 * tanggungan
     + rng.normal(0, 0.7, N))
layak = (rng.random(N) < 1 / (1 + np.exp(-(z - 1.5)))).astype(int)

df = pd.DataFrame({
    "usia": usia, "penghasilan_juta": penghasilan, "tanggungan": tanggungan,
    "lama_kerja_tahun": lama_kerja, "riwayat_kredit": riwayat,
    "pinjaman_juta": pinjaman, "tenor_bulan": tenor,
    "rasio_cicilan": rasio_cicilan, "layak": layak,
})
df.to_csv("dataset_kredit.csv", index=False)
print(df.shape, "proporsi layak:", round(df.layak.mean(), 3))
