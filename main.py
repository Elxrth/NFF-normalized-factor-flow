# ffn.py
# ==================================================================================================
# FFN - Flux Factoriel Normalisé
#
# Version Python / PyTorch
# - GPU CUDA automatique
# - CPU fallback
# - float64 pour les calculs
# - 15 flux mathématiques
# - 7 NUA
# - FFT
# - Signature mathématique
# - ORDER
# - Benchmark de tous les flux
# ==================================================================================================

import math
import time
from dataclasses import dataclass

import torch


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NUM_POINTS = 1_000_000

# Durée logique du flux
TIME_END = 1_000.0

# Nombre de NUA
NUA_COUNT = 7

# Nombre de bins pour l'entropie
ENTROPY_BINS = 256

# float64 = meilleure précision numérique.
# Si tu veux énormément plus de vitesse sur GPU, essaye torch.float32.
DTYPE = torch.float64


# ==================================================================================================
# DEVICE
# ==================================================================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 100)
print("FFN - NORMALIZED FACTORIAL FLOW")
print("=" * 100)

print(f"Device : {DEVICE}")
print(f"Dtype  : {DTYPE}")
print(f"Points : {NUM_POINTS:,}")

if DEVICE.type == "cuda":

    print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print(
        f"VRAM   : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print("=" * 100)


# ==================================================================================================
# DATA STRUCTURES
# ==================================================================================================

@dataclass
class FFNResult:

    name: str

    order: float

    mean: float
    variance: float
    amplitude: float

    variation: float

    autocorrelation: float

    entropy: float

    spectral_entropy: float

    dominant_frequency: float

    zero_crossing_rate: float

    complexity: float

    signature: torch.Tensor

    elapsed: float


# ==================================================================================================
# BASIC UTILITIES
# ==================================================================================================

def synchronize():

    """
    CUDA est asynchrone.
    Cette fonction garantit que les opérations GPU sont terminées
    avant de mesurer le temps.
    """

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def normalize(signal):

    """
    Normalise le signal dans [-1, 1].

    Cela évite que l'amplitude brute influence artificiellement
    la comparaison entre les différents flux.
    """

    minimum = signal.min()
    maximum = signal.max()

    amplitude = maximum - minimum

    if amplitude.abs() < 1e-15:
        return torch.zeros_like(signal)

    return (
        2.0
        *
        (signal - minimum)
        /
        amplitude
        - 1.0
    )


# ==================================================================================================
# NUA 1 - MEAN
# ==================================================================================================

def nua_mean(signal):

    return signal.mean()


# ==================================================================================================
# NUA 2 - VARIANCE
# ==================================================================================================

def nua_variance(signal):

    return signal.var(unbiased=False)


# ==================================================================================================
# NUA 3 - AMPLITUDE
# ==================================================================================================

def nua_amplitude(signal):

    return signal.max() - signal.min()


# ==================================================================================================
# NUA 4 - TEMPORAL VARIATION
# ==================================================================================================

def nua_variation(signal):

    differences = torch.diff(signal)

    return differences.abs().mean()


# ==================================================================================================
# NUA 5 - AUTOCORRELATION
# ==================================================================================================

def nua_autocorrelation(signal):

    x = signal - signal.mean()

    denominator = torch.sum(x * x)

    if denominator.abs() < 1e-15:
        return torch.tensor(
            0.0,
            dtype=signal.dtype,
            device=signal.device
        )

    # Corrélation avec le point précédent.
    numerator = torch.sum(
        x[1:] * x[:-1]
    )

    return torch.clamp(
        numerator / denominator,
        -1.0,
        1.0
    )


# ==================================================================================================
# NUA 6 - SHANNON ENTROPY
# ==================================================================================================

def nua_entropy(signal):

    minimum = signal.min()
    maximum = signal.max()

    if (maximum - minimum).abs() < 1e-15:

        return torch.tensor(
            0.0,
            dtype=signal.dtype,
            device=signal.device
        )

    normalized = (
        (signal - minimum)
        /
        (maximum - minimum)
    )

    bins = torch.clamp(
        (
            normalized
            *
            ENTROPY_BINS
        ).long(),
        0,
        ENTROPY_BINS - 1
    )

    counts = torch.bincount(
        bins,
        minlength=ENTROPY_BINS
    ).to(signal.dtype)

    probabilities = counts / signal.numel()

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -torch.sum(
        probabilities
        *
        torch.log(probabilities)
    )

    max_entropy = math.log(
        ENTROPY_BINS
    )

    return entropy / max_entropy


# ==================================================================================================
# SPECTRAL ANALYSIS
# ==================================================================================================

def spectral_analysis(signal):

    """
    Analyse fréquentielle avec FFT.

    Retourne :
        spectral_entropy
        dominant_frequency
    """

    n = signal.numel()

    centered = signal - signal.mean()

    spectrum = torch.fft.rfft(
        centered
    )

    power = spectrum.abs().square()

    # Retire la composante DC.
    if power.numel() > 1:

        power = power.clone()
        power[0] = 0.0

    total_power = power.sum()

    if total_power <= 1e-15:

        return (
            torch.tensor(
                0.0,
                dtype=signal.dtype,
                device=signal.device
            ),
            0.0
        )

    probabilities = (
        power
        /
        total_power
    )

    nonzero = probabilities > 0

    spectral_entropy = -torch.sum(
        probabilities[nonzero]
        *
        torch.log(
            probabilities[nonzero]
        )
    )

    spectral_entropy /= math.log(
        probabilities.numel()
    )

    dominant_index = torch.argmax(
        power
    )

    # Fréquence normalisée.
    dominant_frequency = (
        dominant_index.item()
        /
        n
    )

    return (
        spectral_entropy,
        dominant_frequency
    )


# ==================================================================================================
# ZERO CROSSING RATE
# ==================================================================================================

def nua_zero_crossing(signal):

    if signal.numel() < 2:
        return 0.0

    left = signal[:-1]
    right = signal[1:]

    crossings = (
        (
            (left < 0)
            &
            (right >= 0)
        )
        |
        (
            (left >= 0)
            &
            (right < 0)
        )
    )

    return crossings.to(
        signal.dtype
    ).mean()


# ==================================================================================================
# COMPLEXITY
# ==================================================================================================

def nua_complexity(
    signal,
    entropy,
    spectral_entropy
):

    """
    Mesure approximative de complexité.

    L'idée :
        entropie temporelle élevée
        +
        entropie spectrale élevée
        =>
        signal moins concentré / plus complexe.
    """

    return (
        entropy
        *
        0.5
        +
        spectral_entropy
        *
        0.5
    )


# ==================================================================================================
# ORDER CALCULATION
# ==================================================================================================

def calculate_order(
    autocorrelation,
    entropy,
    spectral_entropy,
    variation,
    amplitude
):

    # ----------------------------------------------------------------------------------------------
    # Corrélation
    # ----------------------------------------------------------------------------------------------

    correlation_order = (
        autocorrelation + 1.0
    ) / 2.0

    # ----------------------------------------------------------------------------------------------
    # Entropie
    #
    # Plus l'entropie est basse, plus le signal est structuré.
    # ----------------------------------------------------------------------------------------------

    entropy_order = (
        1.0
        -
        torch.clamp(
            entropy,
            0.0,
            1.0
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Entropie spectrale
    #
    # Un signal avec son énergie concentrée sur quelques fréquences
    # est considéré comme plus structuré.
    # ----------------------------------------------------------------------------------------------

    spectral_order = (
        1.0
        -
        torch.clamp(
            spectral_entropy,
            0.0,
            1.0
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Variation normalisée
    # ----------------------------------------------------------------------------------------------

    if amplitude.abs() > 1e-15:

        variation_ratio = (
            variation
            /
            amplitude
        )

        variation_order = (
            1.0
            -
            torch.clamp(
                variation_ratio,
                0.0,
                1.0
            )
        )

    else:

        variation_order = torch.tensor(
            1.0,
            dtype=amplitude.dtype,
            device=amplitude.device
        )

    # ----------------------------------------------------------------------------------------------
    # ORDER
    # ----------------------------------------------------------------------------------------------

    order = (

        correlation_order
        *
        0.30

        +

        entropy_order
        *
        0.20

        +

        spectral_order
        *
        0.35

        +

        variation_order
        *
        0.15

    )

    return order * 100.0


# ==================================================================================================
# SIGNATURE
# ==================================================================================================

def build_signature(
    mean,
    variance,
    amplitude,
    variation,
    autocorrelation,
    entropy,
    spectral_entropy
):

    """
    Signature mathématique du flux.

    Chaque composante est normalisée autant que possible.
    """

    return torch.stack([

        mean,

        variance,

        amplitude,

        variation,

        autocorrelation,

        entropy,

        spectral_entropy,

    ])


# ==================================================================================================
# FLUX 1 - SINE
# ==================================================================================================

def flow_sine(t):

    return torch.sin(
        2.0
        *
        math.pi
        *
        t
    )


# ==================================================================================================
# FLUX 2 - COSINE
# ==================================================================================================

def flow_cosine(t):

    return torch.cos(
        2.0
        *
        math.pi
        *
        t
    )


# ==================================================================================================
# FLUX 3 - QUASI PERIODIC
# ==================================================================================================

def flow_quasi_periodic(t):

    return (
        torch.sin(t)
        +
        torch.sin(
            math.sqrt(2)
            *
            t
        )
    ) / 2.0


# ==================================================================================================
# FLUX 4 - CHIRP
# ==================================================================================================

def flow_chirp(t):

    return torch.sin(
        t * t
    )


# ==================================================================================================
# FLUX 5 - SINC
# ==================================================================================================

def flow_sinc(t):

    return torch.sinc(
        t / math.pi
    )


# ==================================================================================================
# FLUX 6 - DAMPED SINE
# ==================================================================================================

def flow_damped_sine(t):

    return (
        torch.exp(
            -0.001 * t
        )
        *
        torch.sin(t)
    )


# ==================================================================================================
# FLUX 7 - SAWTOOTH
# ==================================================================================================

def flow_sawtooth(t):

    phase = (
        t
        -
        torch.floor(t)
    )

    return (
        2.0
        *
        phase
        -
        1.0
    )


# ==================================================================================================
# FLUX 8 - SQUARE
# ==================================================================================================

def flow_square(t):

    return torch.sign(
        torch.sin(
            2.0
            *
            math.pi
            *
            t
        )
    )


# ==================================================================================================
# FLUX 9 - LOGISTIC MAP
# ==================================================================================================

def flow_logistic(t):

    # Génération vectorisée de la suite logistique.
    #
    # Ici t sert uniquement à déterminer combien de valeurs
    # doivent être produites.

    n = t.numel()

    x = torch.empty(
        n,
        dtype=DTYPE,
        device=DEVICE
    )

    x[0] = 0.371

    r = 4.0

    for i in range(1, n):

        x[i] = (
            r
            *
            x[i - 1]
            *
            (1.0 - x[i - 1])
        )

    return (
        x * 2.0
        -
        1.0
    )


# ==================================================================================================
# FLUX 10 - TENT MAP
# ==================================================================================================

def flow_tent(t):

    n = t.numel()

    x = torch.empty(
        n,
        dtype=DTYPE,
        device=DEVICE
    )

    x[0] = 0.371

    for i in range(1, n):

        previous = x[i - 1]

        x[i] = torch.where(
            previous < 0.5,
            2.0 * previous,
            2.0 * (1.0 - previous)
        )

    return (
        x * 2.0
        -
        1.0
    )


# ==================================================================================================
# FLUX 11 - LORENZ
# ==================================================================================================

def flow_lorenz(t):

    n = t.numel()

    x = torch.empty(
        n,
        dtype=DTYPE,
        device=DEVICE
    )

    y = torch.empty_like(x)
    z = torch.empty_like(x)

    x[0] = 0.1
    y[0] = 0.0
    z[0] = 0.0

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    dt = 0.01

    for i in range(1, n):

        dx = sigma * (
            y[i - 1]
            -
            x[i - 1]
        )

        dy = (
            x[i - 1]
            *
            (
                rho
                -
                z[i - 1]
            )
            -
            y[i - 1]
        )

        dz = (
            x[i - 1]
            *
            y[i - 1]
            -
            beta
            *
            z[i - 1]
        )

        x[i] = (
            x[i - 1]
            +
            dx * dt
        )

        y[i] = (
            y[i - 1]
            +
            dy * dt
        )

        z[i] = (
            z[i - 1]
            +
            dz * dt
        )

    return x


# ==================================================================================================
# FLUX 12 - ROSSLER
# ==================================================================================================

def flow_rossler(t):

    n = t.numel()

    x = torch.empty(
        n,
        dtype=DTYPE,
        device=DEVICE
    )

    y = torch.empty_like(x)
    z = torch.empty_like(x)

    x[0] = 0.1
    y[0] = 0.0
    z[0] = 0.0

    a = 0.2
    b = 0.2
    c = 5.7

    dt = 0.01

    for i in range(1, n):

        dx = (
            -y[i - 1]
            -
            z[i - 1]
        )

        dy = (
            x[i - 1]
            +
            a * y[i - 1]
        )

        dz = (
            b
            +
            z[i - 1]
            *
            (
                x[i - 1]
                -
                c
            )
        )

        x[i] = (
            x[i - 1]
            +
            dx * dt
        )

        y[i] = (
            y[i - 1]
            +
            dy * dt
        )

        z[i] = (
            z[i - 1]
            +
            dz * dt
        )

    return x


# ==================================================================================================
# FLUX 13 - RANDOM WALK
# ==================================================================================================

def flow_random_walk(t):

    random = torch.randn(
        t.numel(),
        dtype=DTYPE,
        device=DEVICE
    )

    return torch.cumsum(
        random,
        dim=0
    )


# ==================================================================================================
# FLUX 14 - BROWNIAN MOTION
# ==================================================================================================

def flow_brownian(t):

    dt = (
        TIME_END
        /
        NUM_POINTS
    )

    noise = torch.randn(
        t.numel(),
        dtype=DTYPE,
        device=DEVICE
    )

    return torch.cumsum(
        noise
        *
        math.sqrt(dt),
        dim=0
    )


# ==================================================================================================
# FLUX 15 - MULTI-SCALE / PINK-LIKE NOISE
# ==================================================================================================

def flow_pink_noise(t):

    n = t.numel()

    result = torch.zeros(
        n,
        dtype=DTYPE,
        device=DEVICE
    )

    # Plusieurs composantes fréquentielles.
    for scale in range(1, 9):

        frequency = (
            2.0 ** scale
        )

        phase = torch.rand(
            1,
            dtype=DTYPE,
            device=DEVICE
        ) * 2.0 * math.pi

        result += (

            torch.sin(
                2.0
                *
                math.pi
                *
                frequency
                *
                t
                +
                phase
            )
            /
            math.sqrt(frequency)

        )

    return result


# ==================================================================================================
# FLUX TABLE
# ==================================================================================================

FLOWS = {

    "Sine": flow_sine,

    "Cosine": flow_cosine,

    "QuasiPeriodic": flow_quasi_periodic,

    "Chirp": flow_chirp,

    "Sinc": flow_sinc,

    "DampedSine": flow_damped_sine,

    "Sawtooth": flow_sawtooth,

    "Square": flow_square,

    "Logistic": flow_logistic,

    "Tent": flow_tent,

    "Lorenz": flow_lorenz,

    "Rossler": flow_rossler,

    "RandomWalk": flow_random_walk,

    "Brownian": flow_brownian,

    "PinkNoise": flow_pink_noise,

}


# ==================================================================================================
# ANALYZE ONE FLOW
# ==================================================================================================

def analyze_flow(
    name,
    generator,
    t
):

    print()
    print("-" * 100)
    print(f"[FFN] Starting: {name}")
    print("-" * 100)

    synchronize()

    start = time.perf_counter()

    # ----------------------------------------------------------------------------------------------
    # Generate
    # ----------------------------------------------------------------------------------------------

    signal = generator(t)

    # ----------------------------------------------------------------------------------------------
    # Nettoyage
    # ----------------------------------------------------------------------------------------------

    signal = torch.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # ----------------------------------------------------------------------------------------------
    # Normalisation
    # ----------------------------------------------------------------------------------------------

    signal = normalize(signal)

    # ----------------------------------------------------------------------------------------------
    # NUA
    # ----------------------------------------------------------------------------------------------

    mean = nua_mean(signal)

    variance = nua_variance(signal)

    amplitude = nua_amplitude(signal)

    variation = nua_variation(signal)

    autocorrelation = nua_autocorrelation(signal)

    entropy = nua_entropy(signal)

    spectral_entropy, dominant_frequency = \
        spectral_analysis(signal)

    zero_crossing = nua_zero_crossing(signal)

    complexity = nua_complexity(
        signal,
        entropy,
        spectral_entropy
    )

    # ----------------------------------------------------------------------------------------------
    # ORDER
    # ----------------------------------------------------------------------------------------------

    order = calculate_order(

        autocorrelation,

        entropy,

        spectral_entropy,

        variation,

        amplitude,

    )

    # ----------------------------------------------------------------------------------------------
    # Signature
    # ----------------------------------------------------------------------------------------------

    signature = build_signature(

        mean,

        variance,

        amplitude,

        variation,

        autocorrelation,

        entropy,

        spectral_entropy,

    )

    synchronize()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    # ----------------------------------------------------------------------------------------------
    # Result
    # ----------------------------------------------------------------------------------------------

    result = FFNResult(

        name=name,

        order=order.item(),

        mean=mean.item(),

        variance=variance.item(),

        amplitude=amplitude.item(),

        variation=variation.item(),

        autocorrelation=autocorrelation.item(),

        entropy=entropy.item(),

        spectral_entropy=spectral_entropy.item(),

        dominant_frequency=dominant_frequency,

        zero_crossing_rate=zero_crossing.item(),

        complexity=complexity.item(),

        signature=signature.detach(),

        elapsed=elapsed,

    )

    # ----------------------------------------------------------------------------------------------
    # Print
    # ----------------------------------------------------------------------------------------------

    print(
        f"ORDER              : {result.order:.8f}"
    )

    print(
        f"Mean               : {result.mean:.8f}"
    )

    print(
        f"Variance           : {result.variance:.8f}"
    )

    print(
        f"Amplitude          : {result.amplitude:.8f}"
    )

    print(
        f"Variation          : {result.variation:.8f}"
    )

    print(
        f"Autocorrelation    : {result.autocorrelation:.8f}"
    )

    print(
        f"Entropy            : {result.entropy:.8f}"
    )

    print(
        f"Spectral Entropy   : {result.spectral_entropy:.8f}"
    )

    print(
        f"Dominant Frequency : {result.dominant_frequency:.8f}"
    )

    print(
        f"Zero Crossing Rate : {result.zero_crossing_rate:.8f}"
    )

    print(
        f"Complexity         : {result.complexity:.8f}"
    )

    print(
        f"Compute Time       : {result.elapsed:.4f}s"
    )

    return result


# ==================================================================================================
# CREATE TIME AXIS
# ==================================================================================================

print()
print("[FFN] Creating time axis...")

start = time.perf_counter()

t = torch.linspace(
    0.0,
    TIME_END,
    NUM_POINTS,
    dtype=DTYPE,
    device=DEVICE
)

synchronize()

print(
    f"[FFN] Time axis ready in "
    f"{time.perf_counter() - start:.4f}s"
)


# ==================================================================================================
# RUN EVERYTHING
# ==================================================================================================

results = []

total_start = time.perf_counter()

for name, generator in FLOWS.items():

    result = analyze_flow(
        name,
        generator,
        t
    )

    results.append(result)


synchronize()

total_time = (
    time.perf_counter()
    -
    total_start
)


# ==================================================================================================
# FINAL RESULTS
# ==================================================================================================

print()
print()
print("=" * 100)
print("FFN - FINAL ORDER RESULTS")
print("=" * 100)

print(
    f"{'#':<4}"
    f"{'FLOW':<20}"
    f"{'ORDER':>14}"
    f"{'ENTROPY':>14}"
    f"{'SPECTRAL':>14}"
    f"{'AUTOCORR':>14}"
)

print("-" * 100)


for i, result in enumerate(
    results,
    start=1
):

    print(

        f"{i:<4}"
        f"{result.name:<20}"
        f"{result.order:>14.6f}"
        f"{result.entropy:>14.6f}"
        f"{result.spectral_entropy:>14.6f}"
        f"{result.autocorrelation:>14.6f}"

    )


print("-" * 100)

print(
    f"Total computation time : {total_time:.4f}s"
)

print(
    f"Points per signal      : {NUM_POINTS:,}"
)

print(
    f"Device                 : {DEVICE}"
)

print("=" * 100)


# ==================================================================================================
# SIGNATURE COMPARISON
# ==================================================================================================

print()
print("=" * 100)
print("FFN - SIGNATURES")
print("=" * 100)

for result in results:

    signature_string = ", ".join(
        f"{value:.5f}"
        for value in result.signature.tolist()
    )

    print(
        f"{result.name:<20} -> [{signature_string}]"
    )


# ==================================================================================================
# FIND MOST ORDERED
# ==================================================================================================

most_ordered = max(
    results,
    key=lambda result: result.order
)

least_ordered = min(
    results,
    key=lambda result: result.order
)


print()
print("=" * 100)

print(
    f"Most ordered : "
    f"{most_ordered.name} "
    f"({most_ordered.order:.6f})"
)

print(
    f"Least ordered: "
    f"{least_ordered.name} "
    f"({least_ordered.order:.6f})"
)

print("=" * 100)


# ==================================================================================================
# REFERENCE TEST
# ==================================================================================================

print()
print("=" * 100)
print("FFN - REFERENCE SIMILARITY")
print("=" * 100)

# On prend le sinus comme référence.
reference = results[0]

reference_signature = (
    reference.signature
)


for result in results:

    distance = torch.linalg.vector_norm(
        result.signature
        -
        reference_signature
    )

    print(
        f"{result.name:<20} "
        f"distance_to_{reference.name:<10} "
        f"= {distance.item():.8f}"
    )


print("=" * 100)
print()
print("FFN analysis completed.")