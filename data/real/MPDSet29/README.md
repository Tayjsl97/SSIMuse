# BMM-Det Real-life Dataset: 29 pairs

This directory contains the 29 query and 29 source melodies used by the
SSIMuse real-data experiment. They were derived from the public
[BMMDet_MPDSet](https://github.com/xuan301/BMMDet_MPDSet) Real-life Dataset,
whose repository is MIT licensed.

The released NPY files are not polyphonic MIDI merges. For every song, one
MIDI note group was selected by a unique note-count and pitch-multiset match
to the authors' cleaned NPY, and its binary melody piano roll was saved as
`uint8`. All 58 saved roll lengths were checked against the selected-track
experiment metadata.

```text
queries/          29 aligned query melody rolls
sources/          29 aligned source melody rolls
metadata.json     source filenames, note counts, origins, and roll lengths
```

Run from the repository root:

```bash
python real_data.py \
  --queries-dir data/real/MPDSet29/queries \
  --sources-dir data/real/MPDSet29/sources \
  --output results/mpdset29.json
```

The correct source is inferred from the matching filename stem: query
`case01.npy` corresponds to source `case01.npy`.

These inputs support the paper's 29-query/29-source protocol. Do not identify
scores obtained from a polyphonic MIDI merge as the aligned-melody result.
