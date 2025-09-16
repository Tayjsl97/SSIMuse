import os
from pathlib import Path
import muspy
import numpy as np
from SSIMuse import SSIMuse_B,SSIMuse_V
import laion_clap
import sys
import json
import random
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
import torchaudio

def silent_load_ckpt(model):
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        model.load_ckpt()
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

def cosine_similarity(a, b):
    a=a.flatten()
    b=b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def pianoroll_to_wav(pianoroll, wav_path):
    pianoroll = pianoroll.astype(np.int32)
    music = muspy.from_pianoroll_representation(
        pianoroll,
        resolution=4,
        program=0,
        is_drum=False,
        encode_velocity=True
    )
    muspy.write_midi("temp.mid", music)
    from midi2audio import FluidSynth
    fs = FluidSynth(sound_font="GeneralUser-GS.sf2")
    fs.midi_to_audio(f"{wav_path}.mid", wav_path)
    if os.path.exists(f"{wav_path}.mid"):
        os.remove(f"{wav_path}.mid")

def compute_CLAP(pr1,pr2):
    embedding1 = CLAP_model.get_audio_embedding_from_filelist([pr1])
    embedding2 = CLAP_model.get_audio_embedding_from_filelist([pr2])
    sim = cosine_similarity(embedding1, embedding2)
    return sim

def get_MERT_emb(audio_path):
    waveform, sampling_rate = torchaudio.load(audio_path)

    target_rate = processor.sampling_rate
    if sampling_rate != target_rate:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    input_audio = waveform.squeeze(0)  # → [time]

    inputs = processor(input_audio, sampling_rate=target_rate, return_tensors="pt")

    with torch.no_grad():
        outputs = MERT_model(**inputs, output_hidden_states=True)

    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

    time_reduced_hidden_states = all_layer_hidden_states.mean(dim=1)

    aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
    weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze(0)

    embedding =  weighted_avg_hidden_states.detach().cpu().numpy()
    return embedding

def compute_MERT(pr1,pr2):
    embedding1 = get_MERT_emb(pr1)
    embedding2 = get_MERT_emb(pr2)
    sim = cosine_similarity(embedding1, embedding2)
    return sim

def baseline_sim(ref_path, mix_path):
    ssimuse_b_class=SSIMuse_B()
    ssimuse_v_class=SSIMuse_V()
    parent_path=Path(ref_path).parent
    results={'lum_b':[],'str_b':[],'ssimuse_b':[],
             'lum_v':[],'con_v':[],'str_v':[],'ssimuse_v':[],
             'CLAP':[],'MERT':[]}
    ref_files=os.listdir(ref_path)
    mix_files=os.listdir(mix_path)
    ref_files_paths=[os.path.join(ref_path,i) for i in ref_files]
    mix_files_paths=[os.path.join(mix_path,i) for i in mix_files]
    save_path=f"{parent_path}/baseline_results.json"
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            results = json.load(f)
    cnt=0
    from itertools import product
    all_pairs = list(product(ref_files_paths, mix_files_paths))
    pairs = random.sample(all_pairs, 4000)
    for i,j in pairs:
        pr1 = np.load(i)
        pr2 = np.load(j)
        lum_b, str_b, ssimuse_b = ssimuse_b_class.compute_ssim(pr1, pr2)
        results['lum_b'].append(lum_b)
        results['str_b'].append(str_b)
        results['ssimuse_b'].append(ssimuse_b)
        lum_v, con_v, str_v, ssimuse_v = ssimuse_v_class.compute_ssim(pr1, pr2)
        results['lum_v'].append(lum_v)
        results['con_v'].append(con_v)
        results['str_v'].append(str_v)
        results['ssimuse_v'].append(ssimuse_v)
        try:
            pianoroll_to_wav(pr1, f"wav/POP909/pr1_{cnt}.wav")
            pianoroll_to_wav(pr2, f"wav/POP909/pr2_{cnt}.wav")
            CLAP_cos = compute_CLAP(f"wav/POP909/pr1_{cnt}.wav",
                                    f"wav/POP909/pr2_{cnt}.wav")
            results['CLAP'].append(CLAP_cos)
            MERT_cos = compute_MERT(f"wav/POP909/pr1_{cnt}.wav",
                                    f"wav/POP909/pr2_{cnt}.wav")
            results['MERT'].append(MERT_cos)
        except Exception as e:
            print("ERROR: ", e)
            continue
        cnt += 1
        if cnt % 10 == 0:
            with open(save_path, 'w') as f:
                json.dump(results, f)
    with open(save_path, 'w') as f:
        json.dump(results, f)

def synthetic_data_sim(ref_path, mix_path, copy_bar=1):
    ssimuse_b_class = SSIMuse_B()
    ssimuse_v_class = SSIMuse_V()
    parent_path = Path(ref_path).parent
    results = {'lum_b': [], 'str_b': [], 'ssimuse_b': [],
               'lum_v': [], 'con_v': [], 'str_v': [], 'ssimuse_v': [],
               'CLAP': [],'MERT': []}
    ref_files = os.listdir(ref_path)
    mix_files = os.listdir(mix_path)
    ref_files_paths = [os.path.join(ref_path, i) for i in ref_files]
    mix_files_paths = [os.path.join(mix_path, i) for i in mix_files]
    save_path = f"{parent_path}/copy_{copy_bar}_results.json"
    copy_len = 16 * copy_bar
    cnt=0
    for i in reversed(ref_files_paths):
        pr1 = np.load(i)
        T,P = pr1.shape
        for _ in range(10):
            start_ref = random.randint(0, T - copy_len)
            segment = pr1[start_ref:start_ref + copy_len]
            mix_idx = random.randint(0, len(mix_files_paths) - 1)
            pr2 = np.load(mix_files_paths[mix_idx])
            start_mix = random.randint(0, T - copy_len)
            pr2[start_mix:start_mix + copy_len]=segment
            lum_b, str_b, ssimuse_b = ssimuse_b_class.compute_ssim(pr1, pr2)
            results['lum_b'].append(lum_b)
            results['str_b'].append(str_b)
            results['ssimuse_b'].append(ssimuse_b)
            lum_v, con_v, str_v, ssimuse_v = ssimuse_v_class.compute_ssim(pr1, pr2)
            results['lum_v'].append(lum_v)
            results['con_v'].append(con_v)
            results['str_v'].append(str_v)
            results['ssimuse_v'].append(ssimuse_v)
            try:
                pianoroll_to_wav(pr1, f"wav/POP909/pr1_bar{copy_bar}_{cnt}_{_}.wav")
                pianoroll_to_wav(pr2, f"wav/POP909/pr2_bar{copy_bar}_{cnt}_{_}.wav")
                CLAP_cos = compute_CLAP(f"wav/POP909/pr1_bar{copy_bar}_{cnt}_{_}.wav",
                                        f"wav/POP909/pr2_bar{copy_bar}_{cnt}_{_}.wav")
                results['CLAP'].append(CLAP_cos)
                MERT_cos = compute_MERT(f"wav/POP909/pr1_bar{copy_bar}_{cnt}_{_}.wav",
                                        f"wav/POP909/pr2_bar{copy_bar}_{cnt}_{_}.wav")
                results['MERT'].append(MERT_cos)
            except Exception as e:
                print("ERROR: ", e)
                continue
        with open(save_path, 'w') as f:
            json.dump(results, f)
        cnt+=1

if __name__ == '__main__':
    local_dir = "MERT-v1-95M"
    MERT_model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(local_dir, trust_remote_code=True)

    CLAP_model = laion_clap.CLAP_Module(enable_fusion=False)
    silent_load_ckpt(CLAP_model)

    reference_path="data/POP909/reference_crops"
    mixture_path="data/POP909/mixture_crops"
    baseline_sim(reference_path, mixture_path)
    synthetic_data_sim(reference_path, mixture_path, copy_bar=1)
