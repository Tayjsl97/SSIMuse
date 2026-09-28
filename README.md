# SSIMuse

SSIMuse measures data replication between symbolic-music pieces represented as
binary piano rolls. It adapts the Structural Similarity Index Measure (SSIM)
to music by combining note-density consistency with local note-event overlap.

The method is training-free and searches for matching material under:

- temporal displacement;
- pitch-class transposition with octave equivalence through folding; and
- uniform duration scaling with musically meaningful ratios.

This repository contains the core metric, controlled-copy evaluation,
real-data source retrieval, MIDI preprocessing, and compact processed data.

## 1. Repository structure

```text
SSIMuse.py          Core SSIMuse metric and FFT-accelerated alignment
controlled_data.py  Controlled-copy evaluation on 16-bar clips
real_data.py        Windowed real-data source retrieval
midi_utils.py       MIDI-to-piano-roll conversion
data/
├── controlled/
│   ├── POP909/     Included polyphonic and melody crops
│   └── Pop1K7/     Data-source and preparation information
└── real/
    ├── MPDSet29/   Included 29-query/29-source melody inputs
    └── CopyrightCases40/  Included 18-query/39-source melody inputs
```

## 2. Installation

SSIMuse requires Python 3.10 or newer.

```bash
git clone https://github.com/Tayjsl97/SSIMuse.git
cd SSIMuse
pip install numpy scipy
```

Install `mido` only if you want to convert MIDI files directly:

```bash
pip install mido
```

No model weights, GPU, or training procedure are required.

## 3. Input piano-roll format

The core metric accepts a NumPy array with shape:

```text
(number_of_time_steps, number_of_MIDI_pitches)
```

The experiments use 128 MIDI-pitch columns, four time steps per quarter note,
and 16 time steps per 4/4 bar. A positive value denotes an active note and zero
denotes inactivity.

## 4. Compare two pieces

```python
import numpy as np
from SSIMuse import SSIMuse

reference = np.load("reference.npy", allow_pickle=False)
candidate = np.load("candidate.npy", allow_pickle=False)

metric = SSIMuse(weight_power=1.0)
result = metric.score(reference, candidate)

print("Density:", result.density)
print("Structure:", result.structure)
print("SSIMuse:", result.total)
```

The first argument is the reference. The second argument is the candidate that
is scaled, shifted in time, and transposed during alignment.

The returned object contains:

| Field | Meaning |
|---|---|
| `density` | Note-density consistency |
| `structure` | Best aligned local note-event overlap |
| `total` | `density × structure` |
| `scale_scores` | Structural score at every tested duration ratio |

The earlier tuple interface is also available:

```python
density, structure, total = metric.compute_ssim(reference, candidate)
```

## 5. Optional core configuration

The following are optional arguments of the `SSIMuse(...)` constructor. 

```python
from SSIMuse import SSIMuse

metric = SSIMuse(
    local_window_steps=16,
    local_hop_steps=16,
    weight_power=1.0,
    time_penalty=0.5,
    scale_penalty=0.125,
)
```

`local_window_steps` and `local_hop_steps` control the local MSSIM comparison.
`weight_power` controls local-score aggregation. `time_penalty` and
`scale_penalty` suppress matches that require large transformations.

Use `weight_power=1.0` for controlled-copy evaluation and
`weight_power=0.0` for the real-data retrieval configuration.

## 6. Convert MIDI to a piano roll

```python
from midi_utils import midi_to_piano_roll

roll = midi_to_piano_roll(
    "song.mid",
    steps_per_quarter=4,
    track_index=2,
    ignore_drums=True,
)
```

`track_index` selects one Standard MIDI File track. If it is omitted, all
pitched tracks are merged. For exact reproduction of the 29-case experiment,
use the included aligned NPY files rather than selecting MIDI tracks again.

## 7. Run the controlled-data evaluation

Run POP909 with polyphonic piano rolls:

```bash
python controlled_data.py \
  --references data/controlled/POP909/reference_crops \
  --mixtures data/controlled/POP909/mixture_crops \
  --bars 1 2 4 8 \
  --output results/pop909_controlled.json
```

Run POP909 with melody piano rolls:

```bash
python controlled_data.py \
  --references data/controlled/POP909/reference_crops_melody \
  --mixtures data/controlled/POP909/mixture_crops_melody \
  --bars 1 2 4 8 \
  --output results/pop909_melody_controlled.json
```

AILabs/Pop1K7 is not bundled in this repository. Download the original
[Pop1K7.zip (302.7 MB)](https://zenodo.org/records/13167761/files/Pop1K7.zip?download=1)
from the official Zenodo record before preparing its controlled clips.

The controlled clips are 16 bars long. `--bars` specifies the copied excerpt
lengths and may contain any integer from 1 to 16.

## 8. Run the real-data retrieval

The included MPDSet29 data uses matching filenames for positive pairs, for
example `queries/case01.npy` and `sources/case01.npy`.

```bash
python real_data.py \
  --queries-dir data/real/MPDSet29/queries \
  --sources-dir data/real/MPDSet29/sources \
  --output results/mpdset29.json
```

Run the included infringement subset of the 40-case copyright dataset:

```bash
python real_data.py \
  --queries-dir data/real/CopyrightCases40/queries \
  --sources-dir data/real/CopyrightCases40/sources \
  --output results/copyright_cases40.json
```

The original data are available from the authors'
[music-copyright-expanded repository](https://github.com/comp-music-lab/music-copyright-expanded/tree/main/CopyrightCases).
The included files contain symbolic note data only; no original audio is
redistributed here.

Additional files in `sources/` are treated as distractor candidates. Every
query must have a same-named source so retrieval accuracy can be calculated.

## License and attribution

SSIMuse code is released under the MIT License. See [LICENSE](LICENSE).

Processed POP909 and BMMDet_MPDSet inputs retain their upstream attribution
and license notices in their respective data directories:

- [POP909 Dataset](https://github.com/music-x-lab/POP909-Dataset)
- [BMMDet_MPDSet](https://github.com/xuan301/BMMDet_MPDSet)
- [40 Music Copyright Cases](https://github.com/comp-music-lab/music-copyright-expanded)
