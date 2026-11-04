import csv
import json
from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 22_050
HOP_LENGTH = 512
N_MFCC = 20
N_MELS = 128

# The script reads these four recordings from the current user's Downloads folder.
AUDIO_FILES = [
	Path.home() / "Downloads" / "kushal.ogg",
	Path.home() / "Downloads" / "varsha.ogg",
	Path.home() / "Downloads" / "kushal2.ogg",
	Path.home() / "Downloads" / "varsha2.ogg",
]

OUTPUT_MATRIX = Path(__file__).with_name("voice_features.npy")
OUTPUT_METADATA = Path(__file__).with_name("voice_features.json")
OUTPUT_CSV = Path(__file__).with_name("voice_features.csv")

# These are the eight features selected for comparing the two voices.
SELECTED_FEATURES = [
	"mel_spectrogram_db_7_mean",
	"chroma_vqt_9_mean",
	"spectral_flatness_1_mean",
	"zero_crossing_rate_1_mean",
	"spectral_centroid_1_std",
	"mfcc_2_std",
	"mel_spectrogram_db_22_std",
	"spectral_contrast_7_mean",
]


def extract_features(audio_path: Path) -> tuple[np.ndarray, list[str]]:
	"""Return a fixed-width vector summarizing librosa features for one recording."""
	signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
	if signal.size == 0:
		raise ValueError(f"Audio file is empty: {audio_path}")

	values: list[float] = []
	column_names: list[str] = []

	def add_scalar(name: str, value: float) -> None:
		values.append(float(value))
		column_names.append(name)

	def add_summary(name: str, feature: np.ndarray) -> None:
		"""Append mean and standard deviation for every feature channel."""
		array = np.asarray(feature)
		if np.iscomplexobj(array):
			array = np.abs(array)
		array = array.astype(np.float64, copy=False)
		if array.ndim == 0:
			array = array.reshape(1, 1)
		elif array.ndim == 1:
			array = array.reshape(1, -1)

		finite = np.isfinite(array)
		count = finite.sum(axis=-1)
		safe_values = np.where(finite, array, 0.0)
		means = np.divide(
			safe_values.sum(axis=-1),
			count,
			out=np.full(array.shape[:-1], np.nan, dtype=np.float64),
			where=count > 0,
		)
		centered = np.where(finite, array - means[..., np.newaxis], 0.0)
		variances = np.divide(
			np.square(centered).sum(axis=-1),
			count,
			out=np.full(array.shape[:-1], np.nan, dtype=np.float64),
			where=count > 0,
		)
		standard_deviations = np.sqrt(variances)

		for index, (mean, std) in enumerate(
			zip(means.reshape(-1), standard_deviations.reshape(-1)), start=1
		):
			values.extend((float(mean), float(std)))
			column_names.extend((f"{name}_{index}_mean", f"{name}_{index}_std"))

	# Time-domain features.
	add_scalar("duration_seconds", len(signal) / sr)
	add_summary("rms", librosa.feature.rms(y=signal, hop_length=HOP_LENGTH))
	add_summary(
		"zero_crossing_rate",
		librosa.feature.zero_crossing_rate(y=signal, hop_length=HOP_LENGTH),
	)

	# Spectral and cepstral features.
	mfcc = librosa.feature.mfcc(
		y=signal, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS, hop_length=HOP_LENGTH
	)
	add_summary("mfcc", mfcc)
	add_summary("mfcc_delta", librosa.feature.delta(mfcc, order=1))
	add_summary("mfcc_delta2", librosa.feature.delta(mfcc, order=2))

	mel_power = librosa.feature.melspectrogram(
		y=signal, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
	)
	add_summary(
		"mel_spectrogram_db",
		librosa.power_to_db(mel_power, ref=np.max),
	)
	add_summary("spectral_centroid", librosa.feature.spectral_centroid(y=signal, sr=sr))
	add_summary(
		"spectral_bandwidth",
		librosa.feature.spectral_bandwidth(y=signal, sr=sr),
	)
	add_summary("spectral_contrast", librosa.feature.spectral_contrast(y=signal, sr=sr))
	add_summary("spectral_flatness", librosa.feature.spectral_flatness(y=signal))
	add_summary(
		"spectral_rolloff_85",
		librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=0.85),
	)
	add_summary(
		"spectral_rolloff_95",
		librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=0.95),
	)
	add_summary(
		"spectral_polynomial",
		librosa.feature.poly_features(y=signal, sr=sr, order=2),
	)

	# Harmonic, chroma, and tonal features.
	harmonic, percussive = librosa.effects.hpss(signal)
	add_summary(
		"chroma_stft",
		librosa.feature.chroma_stft(y=signal, sr=sr, hop_length=HOP_LENGTH),
	)
	add_summary(
		"chroma_cqt",
		librosa.feature.chroma_cqt(y=signal, sr=sr, hop_length=HOP_LENGTH),
	)
	add_summary(
		"chroma_cens",
		librosa.feature.chroma_cens(y=signal, sr=sr, hop_length=HOP_LENGTH),
	)
	add_summary(
		"chroma_vqt",
		librosa.feature.chroma_vqt(
			y=signal, sr=sr, hop_length=HOP_LENGTH, intervals="equal"
		),
	)
	add_summary("tonnetz", librosa.feature.tonnetz(y=harmonic, sr=sr))
	add_scalar(
		"harmonic_to_percussive_rms_ratio",
		float(np.sqrt(np.mean(harmonic**2)) / (np.sqrt(np.mean(percussive**2)) + 1e-10)),
	)

	# Rhythm and onset features.
	onset = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=HOP_LENGTH)
	add_summary("onset_strength", onset)
	add_summary(
		"tempogram",
		librosa.feature.tempogram(
			onset_envelope=onset, sr=sr, hop_length=HOP_LENGTH, win_length=128
		),
	)
	add_summary(
		"fourier_tempogram",
		librosa.feature.fourier_tempogram(
			onset_envelope=onset, sr=sr, hop_length=HOP_LENGTH, win_length=128
		),
	)
	tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=HOP_LENGTH)
	add_scalar("tempo_bpm", float(np.asarray(tempo).reshape(-1)[0]))
	_, beat_frames = librosa.beat.beat_track(
		onset_envelope=onset, sr=sr, hop_length=HOP_LENGTH
	)
	add_scalar("beat_count", float(len(beat_frames)))

	# Voice pitch (fundamental frequency) and voicing estimates.
	try:
		f0, voiced_flag, voiced_probability = librosa.pyin(
			signal,
			fmin=librosa.note_to_hz("C2"),
			fmax=librosa.note_to_hz("C6"),
			sr=sr,
			frame_length=2048,
			hop_length=HOP_LENGTH,
		)
		voiced_f0 = f0[np.isfinite(f0)]
		add_scalar("pitch_f0_mean_hz", float(np.mean(voiced_f0)) if voiced_f0.size else np.nan)
		add_scalar("pitch_f0_std_hz", float(np.std(voiced_f0)) if voiced_f0.size else np.nan)
		add_scalar("pitch_f0_min_hz", float(np.min(voiced_f0)) if voiced_f0.size else np.nan)
		add_scalar("pitch_f0_max_hz", float(np.max(voiced_f0)) if voiced_f0.size else np.nan)
		add_scalar("voiced_fraction", float(np.mean(voiced_flag)))
		add_scalar("voiced_probability_mean", float(np.nanmean(voiced_probability)))
	except (ValueError, RuntimeError) as error:
		print(f"Warning: pitch extraction failed for {audio_path.name}: {error}")
		for name in (
			"pitch_f0_mean_hz",
			"pitch_f0_std_hz",
			"pitch_f0_min_hz",
			"pitch_f0_max_hz",
			"voiced_fraction",
			"voiced_probability_mean",
		):
			add_scalar(name, np.nan)

	return np.asarray(values, dtype=np.float32), column_names


def main() -> None:
	missing_files = [path for path in AUDIO_FILES if not path.is_file()]
	if missing_files:
		missing = "\n".join(str(path) for path in missing_files)
		raise FileNotFoundError(f"Could not find these recordings:\n{missing}")

	rows: list[np.ndarray] = []
	reference_columns: list[str] | None = None
	for path in AUDIO_FILES:
		row, columns = extract_features(path)
		if reference_columns is not None and columns != reference_columns:
			raise RuntimeError(f"Feature columns differ for {path.name}")
		reference_columns = columns
		rows.append(row)
		print(f"Extracted {path.name}: {row.size} features")

	matrix = np.vstack(rows)

	# Print the selected feature values by recording name for easy comparison.
	column_indexes = {name: index for index, name in enumerate(reference_columns or [])}
	print("\nSelected voice features:")
	for path, row in zip(AUDIO_FILES, rows):
		print(f"\n{path.name}")
		for feature_name in SELECTED_FEATURES:
			print(f"  {feature_name}: {row[column_indexes[feature_name]]:.4f}")

	np.save(OUTPUT_MATRIX, matrix)
	OUTPUT_METADATA.write_text(
		json.dumps(
			{
				"shape": list(matrix.shape),
				"sample_rate": SAMPLE_RATE,
				"files": [path.name for path in AUDIO_FILES],
				"columns": reference_columns,
			},
			indent=2,
		),
		encoding="utf-8",
	)
	with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["file", *(reference_columns or [])])
		for path, row in zip(AUDIO_FILES, matrix):
			writer.writerow([path.name, *row.tolist()])
	print(f"NumPy feature matrix shape: {matrix.shape}")
	print(f"Saved matrix: {OUTPUT_MATRIX}")
	print(f"Saved feature labels: {OUTPUT_METADATA}")
	print(f"Saved CSV matrix: {OUTPUT_CSV}")


if __name__ == "__main__":
	main()
