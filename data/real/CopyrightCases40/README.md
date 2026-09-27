# 40 music copyright cases

This directory includes the processed symbolic melody inputs used by the
SSIMuse real-data experiment:

- `queries/`: 18 infringement-query piano rolls;
- `sources/`: 39 candidate-source piano rolls;
- `labels.csv`: court-decision labels for all 40 cases; and
- `manifest.csv`: titles and source-file provenance.

Matching stems identify positive pairs. Case 14 is absent from `sources/`
because its available MIDI contains no pitched notes.

The included NPY files contain note timing and pitch data derived from the
authors' melody transcriptions. They do not contain the original recordings.

- [Download the complete upstream repository ZIP](https://github.com/comp-music-lab/music-copyright-expanded/archive/refs/heads/main.zip)
- [Browse `CopyrightCases/MIDI` and `CopyrightCases/Melody`](https://github.com/comp-music-lab/music-copyright-expanded/tree/main/CopyrightCases)
- [Dataset description and usage notice](https://github.com/comp-music-lab/music-copyright-expanded)

The upstream notice cites CC BY 4.0 for the publication and states that its
original audio excerpts are supplied under fair use. Users who download or
redistribute the original recordings must follow that separate restriction.
