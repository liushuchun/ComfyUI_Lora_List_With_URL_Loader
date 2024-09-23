import comfy.sd
import torch
import os
import sys
import hashlib
import requests
import json
from typing import Dict
import folder_paths
from nodes import LoraLoader, VAELoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


def download_file(url, dest):
    """Download file from URL to the destination."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def file_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    if not os.path.exists(file_path):
        return None
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_lora(lora_name, lora_url="", md5_checksum=""):
    """Check if LoRA is available locally, otherwise download it and validate via MD5."""
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not os.path.exists(lora_path):
        if lora_url:
            print(f"{lora_name} not found locally, downloading from {lora_url}...")
            if not download_file(lora_url, lora_path):
                print(f"Failed to download {lora_name}")
                return None
            if md5_checksum and file_md5(lora_path) != md5_checksum:
                print(f"MD5 mismatch for {lora_name}. Downloaded file does not match expected checksum.")
                return None
        else:
            print(f"{lora_name} not found and no URL provided.")
            return None
    return lora_path


class AVLoraLoader(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        inputs = LoraLoader.INPUT_TYPES()
        inputs["optional"] = {
            "lora_override": ("STRING", {"default": "None"}),
            "lora_url": ("STRING", {"default": ""}),  # 新增URL字段
            "md5_checksum": ("STRING", {"default": ""}),  # MD5校验
            "enabled": ("BOOLEAN", {"default": True}),
        }
        return inputs

    CATEGORY = "Art Venture/Loaders"

    def load_lora(self, model, clip, lora_name, *args, lora_override="None", lora_url="", md5_checksum="", enabled=True, **kwargs):
        if not enabled:
            return (model, clip)

        if lora_override != "None":
            lora_name = lora_override

        lora_path = validate_lora(lora_name, lora_url, md5_checksum)
        if not lora_path:
            return (model, clip)

        return super().load_lora(model, clip, lora_name, *args, **kwargs)


class LoraListStacker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK",)
    FUNCTION = "load_list_lora"
    CATEGORY = "Art Venture/Loaders"

    def parse_lora_list(self, data: str):
        """Parse a list of LoRA models from JSON format."""
        data = data.strip()
        if data == "" or data == "[]" or data is None:
            return []

        print(f"Loading lora list: {data}")

        lora_list = json.loads(data)
        if len(lora_list) == 0:
            return []

        available_loras = folder_paths.get_filename_list("loras")

        lora_params = []
        for lora in lora_list:
            lora_name = lora["name"]
            strength_model = lora["strength"]
            strength_clip = lora["strength"]

            if strength_model == 0 and strength_clip == 0:
                continue

            if lora_name not in available_loras:
                print(f"Not found lora {lora_name}, skipping")
                continue

            lora_params.append((lora_name, strength_model, strength_clip))

        return lora_params

    def load_list_lora(self, data, lora_stack=None):
        loras = self.parse_lora_list(data)

        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        return (loras,)


class LoraListUrlLoader(LoraListStacker):

    @classmethod
    def INPUT_TYPES(s):
        lora_files = ["None"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name1": (lora_files,),
                "model_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_name2": (lora_files,),
                "model_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_name3": (lora_files,),
                "model_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")

    def load_list_lora(self, model, clip, lora_name1, model_strength_1, clip_strength_1,
                       lora_name2, model_strength_2, clip_strength_2,
                       lora_name3, model_strength_3, clip_strength_3):
        loras = []

        for lora_name, model_strength, clip_strength in [(lora_name1, model_strength_1, clip_strength_1),
                                                         (lora_name2, model_strength_2, clip_strength_2),
                                                         (lora_name3, model_strength_3, clip_strength_3)]:
            if lora_name != "None":
                lora_path = validate_lora(lora_name)
                if lora_path:
                    loras.append((lora_name, model_strength, clip_strength))

        if len(loras) == 0:
            return (model, clip)

        # Load LoRAs
        return self.load_loras(loras, model, clip)

    def load_loras(self, lora_params, model, clip):
        """Load the LoRA models into the model and clip."""
        for lora_name, strength_model, strength_clip in lora_params:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_file = comfy.utils.load_torch_file(lora_path)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora_file, strength_model, strength_clip)
        return model, clip


NODE_CLASS_MAPPINGS = {
    "LoraListUrlLoader": LoraListUrlLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraListUrlLoader": "Lora URL List Loader",
}
