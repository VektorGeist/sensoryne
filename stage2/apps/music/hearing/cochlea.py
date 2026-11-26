"""stage2/apps/music/hearing/cochlea.py - cochlea and analyzer.

Sensoryne's own hearing for the music application, built from
scratch with ordinary libraries only (numpy, scipy).  Audio is
taken in blocks.  Each block yields a frame percept: time;
bass, mid, and high band energies, each followed by an attack
and decay envelope; loudness; brightness as a spectral
centroid; spectral flux; a fundamental pitch found as the
strongest peak in the pitch band with the peak position
interpolated between bins; a place vector over 32 tonotopic
places laid out by log frequency from 20 Hz to 16,000 Hz,
spectral power assigned to places by triangular weighting;
the block's energy; and its spectral flatness.

The cochlea measures only.  It names no notes, judges nothing,
and knows nothing about instruments.
"""

import math
import os

import numpy as np
from scipy.io import wavfile

__all__ = [
    "PLACE_COUNT",
    "PLACE_MIN_HZ",
    "PLACE_MAX_HZ",
    "PLACE_FREQS",
    "BANDS",
    "Cochlea",
    "analyze_audio",
    "load_wav",
]

PLACE_COUNT = 32
PLACE_MIN_HZ = 20.0
PLACE_MAX_HZ = 16000.0
PLACE_FREQS = np.logspace(
    math.log10(PLACE_MIN_HZ), math.log10(PLACE_MAX_HZ), PLACE_COUNT
)

# Band edges in Hz: bass, mid, high.
BANDS = {
    "bass": (20.0, 250.0),
    "mid": (250.0, 2000.0),
    "high": (2000.0, 16000.0),
}

PITCH_BAND_HZ = (50.0, 1000.0)

_EPS = 1e-12


def _place_weight_matrix(freqs):
    """Triangular weighting of FFT bins onto the 32 places.

    Each bin's power is split between the two nearest places
    in log frequency, in proportion to its distance from each.
    Returns an array of shape (n_bins, PLACE_COUNT).
    """
    weights = np.zeros((len(freqs), PLACE_COUNT), dtype=np.float64)
    log_min = math.log(PLACE_MIN_HZ)
    log_span = math.log(PLACE_MAX_HZ) - log_min
    for index, freq in enumerate(freqs):
        if freq < PLACE_MIN_HZ or freq > PLACE_MAX_HZ:
            continue
        position = (math.log(freq) - log_min) / log_span * (PLACE_COUNT - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            weights[index, lower] = 1.0
        else:
            fraction = position - lower
            weights[index, lower] = 1.0 - fraction
            weights[index, upper] = fraction
    return weights


def _parabolic_peak(magnitudes, index):
    """Interpolated peak position between bins, in bin units."""
    if index <= 0 or index >= len(magnitudes) - 1:
        return float(index)
    alpha = float(magnitudes[index - 1])
    beta = float(magnitudes[index])
    gamma = float(magnitudes[index + 1])
    denominator = alpha - 2.0 * beta + gamma
    if denominator == 0.0:
        return float(index)
    offset = 0.5 * (alpha - gamma) / denominator
    if not math.isfinite(offset) or abs(offset) > 1.0:
        return float(index)
    return float(index) + offset


def _spectral_flatness(power):
    power = np.asarray(power, dtype=np.float64)
    if power.size == 0:
        return 0.0
    mean_power = float(np.mean(power))
    if mean_power <= _EPS:
        return 0.0
    geometric = float(np.exp(np.mean(np.log(power + _EPS))))
    flatness = geometric / mean_power
    if not math.isfinite(flatness):
        return 0.0
    return float(min(1.0, max(0.0, flatness)))


class Cochlea:
    """Block analyzer over a waveform.

    Parameters
    ----------
    sample_rate: samples per second of the incoming audio.
    fft_size: analysis block length in samples.
    hop_size: advance between blocks in samples.
    """

    def __init__(self, sample_rate=44100, fft_size=4096, hop_size=1024):
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.fft_size <= 0 or self.hop_size <= 0:
            raise ValueError("fft_size and hop_size must be positive")
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)
        self._place_weights = _place_weight_matrix(self.freqs)
        self._window = np.hanning(self.fft_size)
        self._band_masks = {}
        for name, (low, high) in BANDS.items():
            self._band_masks[name] = (self.freqs >= low) & (self.freqs < high)
        pitch_low, pitch_high = PITCH_BAND_HZ
        self._pitch_mask = (self.freqs >= pitch_low) & (self.freqs <= pitch_high)
        if not np.any(self._pitch_mask):
            raise ValueError("pitch band contains no FFT bins")

    def _band_energy(self, power, mask):
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask]))

    def _pitch(self, magnitudes):
        masked = np.where(self._pitch_mask, magnitudes, -1.0)
        peak_index = int(np.argmax(masked))
        if magnitudes[peak_index] <= _EPS:
            return 0.0
        interpolated = _parabolic_peak(magnitudes, peak_index)
        bin_hz = self.sample_rate / float(self.fft_size)
        return float(interpolated * bin_hz)

    def analyze(self, samples):
        """Analyze a mono waveform, returning one dict per block.

        ``samples`` may be a numpy array of floats in -1..1 or
        integer samples, which are scaled to -1..1.
        """
        waveform = np.asarray(samples)
        if waveform.ndim > 1:
            waveform = waveform[:, 0]
        waveform = waveform.astype(np.float64)
        if waveform.size == 0:
            return []
        if np.max(np.abs(waveform)) > 1.5:
            waveform = waveform / 32768.0

        n_frames = 1 + max(0, (len(waveform) - self.fft_size) // self.hop_size)
        if len(waveform) < self.fft_size:
            n_frames = 1
        padded = np.zeros(n_frames * self.hop_size + self.fft_size, dtype=np.float64)
        padded[: len(waveform)] = waveform

        frames = []
        previous_magnitudes = None
        envelopes = {name: 0.0 for name in BANDS}
        attack_rate = 0.65
        decay_rate = 0.85

        for index in range(n_frames):
            start = index * self.hop_size
            block = padded[start:start + self.fft_size]
            windowed = block * self._window
            spectrum = np.fft.rfft(windowed)
            magnitudes = np.abs(spectrum)
            power = magnitudes ** 2

            band_energies = {
                name: self._band_energy(power, mask)
                for name, mask in self._band_masks.items()
            }
            band_attack = {}
            band_decay = {}
            for name, energy in band_energies.items():
                previous = envelopes[name]
                if energy >= previous:
                    envelopes[name] = attack_rate * energy + (1.0 - attack_rate) * previous
                    band_attack[name] = float(envelopes[name])
                    band_decay[name] = float(previous)
                else:
                    envelopes[name] = decay_rate * energy + (1.0 - decay_rate) * previous
                    band_attack[name] = float(previous)
                    band_decay[name] = float(envelopes[name])

            rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
            loudness = float(20.0 * math.log10(rms + _EPS))
            total_power = float(np.sum(power))
            if total_power > _EPS:
                centroid = float(np.sum(self.freqs * magnitudes) / (np.sum(magnitudes) + _EPS))
            else:
                centroid = 0.0

            if previous_magnitudes is None:
                flux = 0.0
            else:
                differences = magnitudes - previous_magnitudes
                flux = float(np.sum(np.maximum(differences, 0.0)))
            previous_magnitudes = magnitudes

            f0 = self._pitch(magnitudes)
            place = self._place_weights.T @ power
            place_total = float(np.sum(place))
            if place_total > _EPS:
                place = place / place_total
            energy = float(np.sum(block ** 2)) / float(len(block)) if len(block) else 0.0
            flatness = _spectral_flatness(power)

            time = float(start) / float(self.sample_rate)
            frames.append(
                {
                    "time": time,
                    "bass": float(band_energies["bass"]),
                    "mid": float(band_energies["mid"]),
                    "high": float(band_energies["high"]),
                    "bass_attack": float(band_attack["bass"]),
                    "bass_decay": float(band_decay["bass"]),
                    "mid_attack": float(band_attack["mid"]),
                    "mid_decay": float(band_decay["mid"]),
                    "high_attack": float(band_attack["high"]),
                    "high_decay": float(band_decay["high"]),
                    "loud": loudness,
                    "loudness": loudness,
                    "centroid": centroid,
                    "brightness": centroid,
                    "flux": flux,
                    "f0": float(f0),
                    "pitch": float(f0),
                    "place": [float(value) for value in place],
                    "energy": energy,
                    "flatness": flatness,
                }
            )
        return frames


def analyze_audio(samples, sample_rate=44100, fft_size=4096, hop_size=1024):
    """Convenience entry point: analyze a waveform's blocks."""
    return Cochlea(sample_rate, fft_size, hop_size).analyze(samples)


def load_wav(path):
    """Load a WAV file, returning (samples_as_float, sample_rate)."""
    if not os.path.exists(path):
        raise FileNotFoundError("wav file not found: %s" % path)
    sample_rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        samples = data.astype(np.float64) / float(max(abs(info.min), info.max))
    else:
        samples = data.astype(np.float64)
    return samples, int(sample_rate)
