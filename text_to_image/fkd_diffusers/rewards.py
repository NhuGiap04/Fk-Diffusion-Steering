import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import hpsv2
from transformers import AutoModel, AutoProcessor

from image_reward_utils import rm_load
from llm_grading import LLMGrader

HPS_REWARD_NAMES = {"HPS", "HPSv2", "HumanPreference"}
CLIP_REWARD_NAMES = {"Clip-Score", "CLIP-Score"}
PICKSCORE_PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_MODEL_NAME = "yuvalkirstain/PickScore_v1"
AESTHETIC_MODEL_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
    "sac+logos+ava1-l14-linearMSE.pth"
)

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
    "PickScore": None,
    "PickScoreProcessor": None,
    "Aesthetic": None,
}


# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score"):
    if reward_name != "LLMGrader":
        print("`metric_to_chase` will be ignored as it only applies to 'LLMGrader' as the `reward_name`")
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)
    
    elif reward_name in CLIP_REWARD_NAMES:
        return do_clip_score(images=images, prompts=prompts)
    
    elif reward_name in HPS_REWARD_NAMES:
        return do_human_preference_score(images=images, prompts=prompts)

    elif reward_name == "PickScore":
        return do_pick_score(images=images, prompts=prompts)

    elif reward_name == "Aesthetic":
        return do_aesthetic_score(images=images)

    elif reward_name == "LLMGrader":
        return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase)
    
    else:
        raise ValueError(f"Unknown metric: {reward_name}")
    
# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    if use_paths:
        scores = hpsv2.score(images, prompts, hps_version="v2.1")
        scores = [float(score) for score in scores]
    else:
        scores = []
        for i, image in enumerate(images):
            score = hpsv2.score(image, prompts[i], hps_version="v2.1")
            # print(f"Human preference score for image {i}: {score}")
            score = float(score[0])
            scores.append(score)

    # print(f"Human preference scores: {scores}")
    return scores


# Compute PickScore
def do_pick_score(*, images, prompts):
    global REWARDS_DICT
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if REWARDS_DICT["PickScoreProcessor"] is None:
        REWARDS_DICT["PickScoreProcessor"] = AutoProcessor.from_pretrained(
            PICKSCORE_PROCESSOR_NAME
        )
    if REWARDS_DICT["PickScore"] is None:
        REWARDS_DICT["PickScore"] = AutoModel.from_pretrained(
            PICKSCORE_MODEL_NAME
        ).eval().to(device)

    processor = REWARDS_DICT["PickScoreProcessor"]
    model = REWARDS_DICT["PickScore"]

    with torch.no_grad():
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        text_inputs = processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores = model.logit_scale.exp() * (text_features * image_features).sum(dim=-1)

    return scores.detach().cpu().tolist()


class AestheticMLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Compute LAION aesthetic score
def do_aesthetic_score(*, images):
    global REWARDS_DICT
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device=device)
    if REWARDS_DICT["Aesthetic"] is None:
        model = AestheticMLP()
        state_dict = torch.hub.load_state_dict_from_url(
            AESTHETIC_MODEL_URL,
            map_location=device,
            file_name="sac+logos+ava1-l14-linearMSE.pth",
        )
        model.load_state_dict(state_dict)
        REWARDS_DICT["Aesthetic"] = model.eval().to(device)

    clip_score = REWARDS_DICT["Clip-Score"]
    aesthetic_model = REWARDS_DICT["Aesthetic"]

    with torch.no_grad():
        image_batch = torch.cat(
            [
                clip_score.preprocess(image).unsqueeze(0).to(device)
                for image in images
            ],
            dim=0,
        )
        image_features = clip_score.clip_model.encode_image(image_batch).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = aesthetic_model(image_features).squeeze(-1)

    return scores.detach().cpu().tolist()

# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        arr_clip_result = []
        arr_img_features = []
        for i, prompt in enumerate(prompts):
            clip_result, feature_vect = REWARDS_DICT["Clip-Score"].score(
                prompt, images[i], return_feature=True
            )

            arr_clip_result.append(clip_result.item())
            arr_img_features.append(feature_vect['image'])

    # calculate diversity by computing pairwise similarity between image features
    diversity = torch.zeros(len(images), len(images))
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
            diversity[j, i] = diversity[i, j]
    n_samples = len(images)
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()

# Compute ImageReward
def do_image_reward(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    with torch.no_grad():
        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
        # image_reward_result = [REWARDS_DICT["ImageReward"].score(prompt, images[i]) for i, prompt in enumerate(prompts)]

    return image_reward_result

# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        clip_result = [
            REWARDS_DICT["Clip-Score"].score(prompt, images[i])
            for i, prompt in enumerate(prompts)
        ]
    return clip_result


# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
    global REWARDS_DICT
    
    if REWARDS_DICT["LLMGrader"] is None:
        REWARDS_DICT["LLMGrader"]  = LLMGrader()
    llm_grading_result = [
        REWARDS_DICT["LLMGrader"].score(images=images[i], prompts=prompt, metric_to_chase=metric_to_chase)
        for i, prompt in enumerate(prompts)
    ]
    return llm_grading_result


'''
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards

        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}

        return rewards.detach().cpu().numpy().item()
