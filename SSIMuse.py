import torch.nn as nn
from fastdtw import fastdtw
import numpy as np

class SSIMuse_B(nn.Module):

    def __init__(self, max_time_shift=32, max_pitch_shift=64):
        super().__init__()
        self.L = 1
        self.C1 = (0.01 * self.L) ** 2
        self.C2 = (0.03 * self.L) ** 2
        self.C3 = self.C2 / 2
        self.max_time_shift = max_time_shift
        self.max_pitch_shift = max_pitch_shift

    def compute_luminance(self,pr1,pr2):
        pr1 = np.where(pr1 > 0, 1, 0)
        pr2 = np.where(pr2 > 0, 1, 0)
        pr1 = (pr1 > 0).astype(np.uint8)
        pr2 = (pr2 > 0).astype(np.uint8)
        T, P = pr1.shape

        lum_scores = []
        for start in range(0, T, 16):
            end = start + 16
            if end > T:
                break
            w1 = pr1[start:end]
            w2 = pr2[start:end]
            mu_x = w1.mean()
            mu_y = w2.mean()
            luminance = (2 * mu_x * mu_y + self.C1) / (mu_x ** 2 + mu_y ** 2 + self.C1)
            lum_scores.append(luminance)
        luminance = np.mean(lum_scores)
        return luminance

    def pr2scale(self,pr):
        pr_scale = np.zeros((pr.shape[0],12))
        for i in range(pr.shape[0]):
            for j in range(pr.shape[1]):
                if pr[i][j]!=0:
                    pr_scale[i][j%12]+=1
        return pr_scale

    def compute_structure(self, pr1, pr2):
        pr1 = np.where(pr1 > 0, 1, 0)
        pr2 = np.where(pr2 > 0, 1, 0)
        pr1 = (pr1 > 0).astype(np.uint8)
        pr2 = (pr2 > 0).astype(np.uint8)
        pr1 = self.pr2scale(pr1)
        pr2 = self.pr2scale(pr2)

        assert pr1.ndim == 2 and pr2.ndim == 2

        T, P = pr1.shape
        self.max_time_shift = T

        best_structure = None

        for time_shift in range(-self.max_time_shift//2, self.max_time_shift//2 + 1):
            if time_shift != 0:
                pr2_time = np.roll(pr2, shift=time_shift, axis=0)
            else:
                pr2_time = pr2
            for pitch_shift in range(-5, 7):
                pr2_shifted = np.roll(pr2_time, shift=pitch_shift, axis=1)

                mask = (pr1 > 0) & (pr2_shifted > 0)
                if mask.sum() == 0:
                    continue

                jaccard_scores = []
                for start in range(0, T, 16):
                    end = start + 16
                    if end > T:
                        break
                    w1 = pr1[start:end]
                    w2 = pr2_shifted[start:end]
                    if np.sum(w1) == 0 and np.sum(w2) == 0:
                        continue
                    inter = np.minimum(w1, w2)
                    union = np.maximum(w1, w2)
                    jaccard = inter.sum() / (union.sum() + 1e-8)
                    jaccard_scores.append(jaccard)

                weights = [s for s in jaccard_scores]  #math.sqrt(s), s, s ** 2
                jaccard = sum(score * weight for score, weight in zip(jaccard_scores, weights)) / sum(weights)

                wrapped_fraction = abs(time_shift) / T
                penalty = 1.0 - 0.5 * wrapped_fraction
                jaccard=jaccard*penalty

                structure = jaccard
                if best_structure is None:
                    best_structure = structure
                else:
                    if structure > best_structure:
                        best_structure = structure
        return best_structure

    def compute_ssim(self,pr1,pr2):
        lum = self.compute_luminance(pr1, pr2)
        struct = self.compute_structure(pr1, pr2)
        ssimuse_b = lum * struct
        return lum,struct,ssimuse_b


class SSIMuse_V(nn.Module):

    def __init__(self):
        super().__init__()
        self.L = 127
        self.C1 = (0.01 * self.L) ** 2
        self.C2 = (0.03 * self.L) ** 2
        self.C3 = self.C2 / 2

    def compute_mu(self,pr):
        nonzero_values = pr[pr != 0]
        if nonzero_values.size == 0:
            return 0.0
        return np.mean(nonzero_values)

    def compute_sigma(self,pr):
        nonzero_values = pr[pr != 0]
        if nonzero_values.size == 0:
            return 0.0
        return np.std(nonzero_values)

    def compute_luminance(self, pr1, pr2):
        mu_x = self.compute_mu(pr1)
        mu_y = self.compute_mu(pr2)
        luminance = (2 * mu_x * mu_y + self.C1) / (mu_x ** 2 + mu_y ** 2 + self.C1)
        return luminance

    def compute_contrast(self, pr1, pr2):
        sigma_x = self.compute_sigma(pr1)
        sigma_y = self.compute_sigma(pr2)
        contrast = (2 * sigma_x * sigma_y + self.C2) / (sigma_x ** 2 + sigma_y ** 2 + self.C2)
        return contrast

    def extract_velocity_profiles(self,pr, type="max"):
        max_vels = []
        mean_vels = []

        for t in range(pr.shape[0]):
            frame = pr[t, :]
            nonzero = frame[frame > 0]
            if nonzero.size > 0:
                max_vels.append(np.max(nonzero))
                mean_vels.append(np.mean(nonzero))
        if type == "max":
            return np.array(max_vels)
        else:
            return np.array(mean_vels)

    def compute_structure(self, pr1, pr2):
        seq1=self.extract_velocity_profiles(pr1, type="max")
        seq2=self.extract_velocity_profiles(pr2, type="max")
        try:
            distance, path = fastdtw(seq1, seq2, dist=lambda x, y: abs(x - y))
        except Exception as e:
            print(f"ERROR: {e}")
            return None
        aligned1 = np.array([seq1[i] for i, _ in path])
        aligned2 = np.array([seq2[j] for _, j in path])
        sigma_x = np.std(aligned1)
        sigma_y = np.std(aligned2)
        sigma_xy=np.cov(aligned1, aligned2)[0, 1]

        structure = (sigma_xy + self.C3) / (sigma_x * sigma_y + self.C3)
        return structure

    def compute_ssim(self, pr1, pr2):
        luminance = self.compute_luminance(pr1, pr2)
        contrast = self.compute_contrast(pr1, pr2)
        structure = self.compute_structure(pr1, pr2)
        ssimuse_v = luminance * contrast * structure
        return luminance,contrast,structure,ssimuse_v
