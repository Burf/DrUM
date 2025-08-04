import os
    
import torch

from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .model import DrUM as backbone
from .sampling import coreset_sampling

def stable_diffusion(large):
    """
    openai/clip-vit-large-patch14, CLIPTextModel, skip -1
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.3, skip = -1,  batch_size = 64, **kwargs):
        return large(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, **kwargs), None
    return inference

def stable_diffusion_v2(huge):
    """
    openai/clip-vit-huge-patch14, CLIPTextModel, skip -1
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.3, skip = -1,  batch_size = 64, **kwargs):
        return huge(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, **kwargs), None
    return inference

def stable_diffusion_xl(large, bigG):
    """
    openai/clip-vit-large-patch14, CLIPTextModel, skip -2, unnorm
    laion/CLIP-ViT-bigG-14-laion2B-39B-b160k, CLIPTextModelWithProjection, skip -2, unnorm, pooling + proj
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.3, skip = -2, batch_size = 64, **kwargs):
        hidden_state = large(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, normalize = False, **kwargs)
        if skip == -1:
            hidden_state2, pool_hidden_state = bigG(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, normalize = False, normalize_pool = True, **kwargs)
        else:
            hidden_state2 = bigG(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, normalize = False, **kwargs)
            pool_hidden_state = bigG(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, skip = -1, batch_size = batch_size, normalize = False, normalize_pool = True, **kwargs)[1]
        hidden_state = torch.cat([hidden_state, hidden_state2], dim = -1)
        pool_hidden_state = bigG.projection_text_hidden_state(pool_hidden_state)
        return hidden_state.type(pool_hidden_state.dtype), pool_hidden_state
    return inference

def stable_diffusion_v3(large, bigG, t5):
    """
    openai/clip-vit-large-patch14, CLIPTextModelWithProjection, skip -2, unnorm, pooling + proj
    laion/CLIP-ViT-bigG-14-laion2B-39B-b160k, CLIPTextModelWithProjection, skip -2, unnorm, pooling + proj
    t5-v1_1-xxl, T5EncoderModel
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.3, skip = -2, batch_size = 64, **kwargs):
        if skip == -1:
            hidden_state, pool_hidden_state = large(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, normalize = False, normalize_pool = True, **kwargs)
            hidden_state2, pool_hidden_state2 = bigG(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, normalize = False, normalize_pool = True, **kwargs)
        else:
            hidden_state = large(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, normalize = False, **kwargs)
            hidden_state2 = bigG(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, skip = skip, batch_size = batch_size, normalize = False, **kwargs)
            pool_hidden_state = large(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, skip = -1, batch_size = batch_size, normalize = False, normalize_pool = True, **kwargs)[1]
            pool_hidden_state2 = bigG(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, skip = -1, batch_size = batch_size, normalize = False, normalize_pool = True, **kwargs)[1]
        hidden_state3 = t5(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, batch_size = batch_size, normalize = False, **kwargs)
        hidden_state = torch.cat([hidden_state, hidden_state2], dim = -1)
        pool_hidden_state = large.projection_text_hidden_state(pool_hidden_state)
        pool_hidden_state2 = bigG.projection_text_hidden_state(pool_hidden_state2)
        hidden_state = torch.nn.functional.pad(hidden_state, (0, hidden_state3.shape[-1] - hidden_state.shape[-1]))
        hidden_state = torch.cat([hidden_state, hidden_state3], dim = -2)
        pool_hidden_state = torch.cat([pool_hidden_state, pool_hidden_state2], dim = -1)
        return hidden_state.type(pool_hidden_state.dtype), pool_hidden_state
    return inference

def flux(large, t5):
    """
    openai/clip-vit-large-patch14, CLIPTextModel, pooling
    t5-v1_1-xxl, T5EncoderModel
    """
    def inference(prompt, ref_prompt = None, weight = None, alpha = 0.3, skip = None, batch_size = 64, **kwargs):
        hidden_state = t5(prompt, ref_prompt, pooling = False, weight = weight, alpha = alpha, batch_size = batch_size, normalize = False, **kwargs)
        pool_hidden_state = large(prompt, ref_prompt, pooling = True, weight = weight, alpha = alpha, skip = -1, batch_size = batch_size, normalize = False, normalize_pool = True, **kwargs)[1]
        return hidden_state.type(pool_hidden_state.dtype), pool_hidden_state
    return inference

def peca(pipeline, save_path = "./weight", n_layer = 10):
    if os.path.exists(os.path.join(save_path, "L.pth")) or os.path.exists(os.path.join(save_path, "H.pth")):
        load_func = torch.load
        postfix = "pth"
    else:
        from safetensors.torch import load_file as load_func
        postfix = "safetensors"
    
    if "flux" in pipeline.config._name_or_path.split("/")[-1].lower():
        model = pipeline.text_encoder
        processor = pipeline.tokenizer
        model2 = pipeline.text_encoder_2
        processor2 = pipeline.tokenizer_2

        large = backbone(model, processor, n_layer = n_layer, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        large.adapter.load_state_dict(load_func(os.path.join(save_path, "L.{0}".format(postfix))))
        t5 = backbone(model2, processor2, n_layer = n_layer, encode_ratio = 4, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        t5.adapter.load_state_dict(load_func(os.path.join(save_path, "T5.{0}".format(postfix))))
        empty, pool = large.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        large.adapter.set_base_query(torch.cat([empty, pool.unsqueeze(1)], dim = 1))
        empty, pool = t5.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        t5.adapter.set_base_query(empty)

        feature_encoder = large
        encoder = flux(large, t5)
        size = 1024
        num_inference_steps = 28
        skip = -2
    elif "stable-diffusion-3.5" in pipeline.config._name_or_path.split("/")[-1].lower(): #sd v3
        model = pipeline.text_encoder
        processor = pipeline.tokenizer
        model2 = pipeline.text_encoder_2
        processor2 = pipeline.tokenizer_2
        model3 = pipeline.text_encoder_3
        processor3 = pipeline.tokenizer_3

        large = backbone(model, processor, n_layer = n_layer, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        large.adapter.load_state_dict(load_func(os.path.join(save_path, "L.{0}".format(postfix))))
        bigG = backbone(model2, processor2, n_layer = n_layer, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        bigG.adapter.load_state_dict(load_func(os.path.join(save_path, "bigG.{0}".format(postfix))))
        t5 = backbone(model3, processor3, n_layer = n_layer, encode_ratio = 4, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        t5.adapter.load_state_dict(load_func(os.path.join(save_path, "T5.{0}".format(postfix))))
        empty, pool = large.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        large.adapter.set_base_query(torch.cat([empty, pool.unsqueeze(1)], dim = 1))
        empty, pool = bigG.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        bigG.adapter.set_base_query(torch.cat([empty, pool.unsqueeze(1)], dim = 1))
        empty, pool = t5.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        t5.adapter.set_base_query(empty)

        feature_encoder = large
        encoder = stable_diffusion_v3(large, bigG, t5)
        size = 1024
        num_inference_steps = 28
        skip = -2
    elif "xl-base" in pipeline.config._name_or_path.split("/")[-1].lower(): #sd xl
        model = pipeline.text_encoder
        processor = pipeline.tokenizer
        model2 = pipeline.text_encoder_2
        processor2 = pipeline.tokenizer_2

        large = backbone(model, processor, n_layer = n_layer, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        large.adapter.load_state_dict(load_func(os.path.join(save_path, "L.{0}".format(postfix))))
        bigG = backbone(model2, processor2, n_layer = n_layer, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        bigG.adapter.load_state_dict(load_func(os.path.join(save_path, "bigG.{0}".format(postfix))))
        empty, pool = large.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        large.adapter.set_base_query(torch.cat([empty, pool.unsqueeze(1)], dim = 1))
        empty, pool = bigG.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        bigG.adapter.set_base_query(torch.cat([empty, pool.unsqueeze(1)], dim = 1))

        feature_encoder = large
        encoder = stable_diffusion_xl(large, bigG)
        size = 1024
        num_inference_steps = 50
        skip = -2
    elif "stable-diffusion-2" in pipeline.config._name_or_path.split("/")[-1].lower():
        model = pipeline.text_encoder
        processor = pipeline.tokenizer

        huge = backbone(model, processor, n_layer = n_layer, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        huge.adapter.load_state_dict(load_func(os.path.join(save_path, "H.{0}".format(postfix))))
        empty, pool = huge.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        huge.adapter.set_base_query(torch.cat([empty, pool.unsqueeze(1)], dim = 1))

        feature_encoder = huge
        encoder = stable_diffusion_v2(huge)
        size = 768
        num_inference_steps = 50
        skip = -1
    else: #sd
        model = pipeline.text_encoder
        processor = pipeline.tokenizer

        large = backbone(model, processor, n_layer = n_layer, pos = False, cls_pos = False, dropout = 0.0).to(pipeline.device).eval()
        large.adapter.load_state_dict(load_func(os.path.join(save_path, "L.{0}".format(postfix))))
        empty, pool = large.encode_prompt("", pooling = True, normalize = False, normalize_pool = False)
        large.adapter.set_base_query(torch.cat([empty, pool.unsqueeze(1)], dim = 1))

        feature_encoder = large
        encoder = stable_diffusion(large)
        size = 512
        num_inference_steps = 50
        skip = -1
    return encoder, feature_encoder.get_text_feature, size, num_inference_steps, skip

class DrUM(DiffusionPipeline):
    def __init__(self, pipeline, repo_id = "Burf/DrUM", weight = None, torch_dtype = torch.bfloat16, device = "cuda"):
        """
        DrUM for various T2I diffusion models
        """
        self.pipeline = pipeline if not isinstance(pipeline, str) else self.load_pipeline(pipeline, torch_dtype = torch_dtype, device = device)
        self.repo_id = repo_id
        
        self.adapter, self.feature_encoder, self.size, self.num_inference_steps, self.skip = self.load_peca(self.pipeline, repo_id, weight)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, repo_id = "Burf/DrUM", torch_dtype = torch.bfloat16, device = "cuda", weight = None):
        """
        Load DrUM adapter with appropriate pipeline
        """
        pipeline = cls.load_pipeline(pretrained_model_name_or_path, torch_dtype, device)
        return cls(pipeline = pipeline, repo_id = repo_id, weight = weight, torch_dtype = torch_dtype, device = device)
    
    @staticmethod
    def load_pipeline(model_id, torch_dtype = torch.bfloat16, device = "cuda"):
        name = model_id.split("/")[-1].lower()
        if "flux" in name:
            pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype = torch_dtype)
        elif "stable-diffusion-3.5" in name:
            pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype = torch_dtype)
        else:
            pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype = torch_dtype)
        
        pipeline = pipeline.to(device if torch.cuda.is_available() else "cpu")
        #pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        return pipeline
    
    def load_weight(self, pipeline, repo_id = "Burf/DrUM", weight = None):
        name = pipeline.config._name_or_path.split("/")[-1].lower()
        
        weights = []
        if "flux" in name:
            weights = ["L.safetensors", "T5.safetensors"]
        elif "stable-diffusion-3.5" in name:
            weights = ["L.safetensors", "bigG.safetensors", "T5.safetensors"]
        elif "xl-base" in name:
            weights = ["L.safetensors", "bigG.safetensors"]
        elif "stable-diffusion-2" in name:
            weights = ["H.safetensors"]
        else:  # SD v1.5
            weights = ["L.safetensors"]
        
        for weight_file in weights:
            if isinstance(weight, str) and os.path.exists(os.path.join(weight, weight_file)):
                weight_path = weight
                break
            else:
                safetensor_path = hf_hub_download(repo_id = repo_id, filename = "weight/" + weight_file)
                weight_path = os.path.dirname(safetensor_path)
        return weight_path
    
    def load_peca(self, pipeline, repo_id = "Burf/DrUM", weight = None):
        adapter, feature_encoder, size, num_inference_steps, skip = peca(pipeline, save_path = self.load_weight(pipeline, repo_id, weight))
        return adapter, feature_encoder, size, num_inference_steps, skip
    
    def __call__(self, prompt, ref = None, weight = None, alpha = 0.3, skip = None, sampling = False, seed = 42, 
                 size = None, num_inference_steps = None, num_images_per_prompt = 1):
        """
        Generate images using DrUM adapter
        
        Args:
            prompt: Text prompt for generation
            ref: Reference prompts (list of strings)
            weight: Weights for reference prompts (list of floats)
            alpha: Personalization strength (0-1)
            skip: Text condition axis
            sampling: Whether to use coreset sampling for reference selection (default: False)
            seed: Random seed
            size: Image size
            num_inference_steps: Inference steps
            num_images_per_prompt: Number of images to generate
            
        Returns:
            Personalized images (list of PIL Images)
        """
        size = self.size if size is None else size
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        skip = self.skip if skip is None else skip
        
        if sampling and isinstance(ref, (tuple, list)) and 1 < len(ref):
            import numpy as np
            
            with torch.no_grad():
                feature = self.feature_encoder(ref).cpu().float().numpy()
            
            indices = coreset_sampling(feature, weight = weight, seed = seed)
            ref = np.array(ref)[indices].tolist()
            
            if isinstance(weight, (tuple, list)) and len(weight) == len(ref):
                weight = np.array(weight)[indices].tolist()
        
        generator = torch.Generator(self.pipeline.device).manual_seed(seed)
        with torch.no_grad():
            cond, pool_cond = self.adapter(prompt, ref, weight = weight, alpha = alpha, skip = skip)
            
            pipe_kwargs = {
                "num_images_per_prompt": num_images_per_prompt,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "height": size,
                "width": size
            }
            
            pipe_kwargs["prompt_embeds"] = cond.type(self.pipeline.dtype)
            if pool_cond is not None:
                pipe_kwargs["pooled_prompt_embeds"] = pool_cond.type(self.pipeline.dtype)
            
            name = self.pipeline.config._name_or_path.split("/")[-1].lower()
            if "flux" in name or "stable-diffusion-3" in name:
                pipe_kwargs["max_sequence_length"] = 256
            
            images = self.pipeline(**pipe_kwargs).images
        return images