# POP909 controlled data

Processed from the [POP909 Dataset](https://github.com/music-x-lab/POP909-Dataset).
The upstream repository publishes POP909 under the MIT License and requests
citation of the POP909 ISMIR 2020 paper.

Each file is a 16-bar piano roll with shape `(256, 128)` and dtype `uint8`.

```text
reference_crops/          400 polyphonic reference clips
mixture_crops/            400 polyphonic mixture clips
reference_crops_melody/   400 melody reference clips
mixture_crops_melody/     400 melody mixture clips
```

Citation:

```bibtex
@inproceedings{wang2020pop909,
  title={POP909: A Pop-song Dataset for Music Arrangement Generation},
  author={Wang, Ziyu and Chen, Ke and Jiang, Junyan and Zhang, Yiyi and Xu, Maoran and Dai, Shuqi and Bin, Guxian and Xia, Gus},
  booktitle={Proceedings of the 21st International Society for Music Information Retrieval Conference},
  year={2020}
}
```
