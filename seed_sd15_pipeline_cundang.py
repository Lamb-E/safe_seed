import os
import torch
import cv2
import time
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from seed_mask import projection_matrix, safree_projection, build_safree_embeddings


class FreePromptPipeline(StableDiffusionPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        # 支持 PIL 或 tensor；送入 VAE 前对齐 device/dtype
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1.0
            image = image.permute(2, 0, 1).unsqueeze(0)

        image = image.to(self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(image).latent_dist.mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = (latents / 0.18215).to(self.vae.device, dtype=self.vae.dtype)
        image = self.vae.decode(latents).sample    # BCHW, [-1,1]
        image = (image / 2 + 0.5).clamp(0, 1)      # [0,1]

        if return_type == 'np':
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            return image
        elif return_type == 'pt':
            return image
        else:
            raise ValueError("return_type must be 'np' or 'pt'")

    def latent2image_grad(self, latents):
        latents = (latents / 0.18215).to(self.vae.device, dtype=self.vae.dtype)
        image = self.vae.decode(latents).sample    # 仍是 [-1,1]（若你需要梯度，别做归一化）
        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        k_invert=1):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        dtype = self.unet.dtype
        text_embeddings = text_embeddings.to(dtype)

        latents = self.generate_with_k_invert_at_t0(
        prompt=prompt,
        batch_size= batch_size,
        height= height,
        width = width,
        num_inference_steps = num_inference_steps,
        guidance_scale =guidance_scale,
        eta= eta,
        latents = None,
        neg_prompt=neg_prompt,
        unconditioning= None,
        k_invert = k_invert,                # <—— 在 t0 进行几次 step→invert
        return_intermediates = False,
        )

        # 验证latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        assert latents.shape == latents_shape, (
            f"The shape of input latent tensor {latents.shape} "
            f"should equal to predefined one {latents_shape}."
        )

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            unconditional_embeddings = unconditional_embeddings.to(dtype)
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)

        latents_list = [latents]
        pred_x0_list = [latents]

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):

            # if i < 7:
            #     guidance_scale = 2
            # else:
            #     print("amazing")
            #     guidance_scale = 15
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict tghe noise
            model_inputs = model_inputs.to(dtype)
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="np") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="np") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        dtype = self.unet.dtype
        text_embeddings = text_embeddings.to(dtype)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        # exit()
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0] 
            unconditional_embeddings = unconditional_embeddings.to(dtype)
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            model_inputs = model_inputs.to(dtype)
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents

    @torch.no_grad()
    def generate_with_k_invert_at_t0(
        self,
        prompt,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        latents: torch.FloatTensor = None,
        neg_prompt: str = None,
        unconditioning=None,
        k_invert: int = 1,                # <—— 在 t0 进行几次 step→invert
        return_intermediates: bool = False,
    ):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 1) 文本编码
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str) and batch_size > 1:
            prompt = [prompt] * batch_size

        if isinstance(neg_prompt, list):
            batch_size = len(neg_prompt)
        elif isinstance(neg_prompt, str) and batch_size > 1:
            neg_prompt = [neg_prompt] * batch_size
        
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        dtype = self.unet.dtype
        text_embeddings = text_embeddings.to(device=DEVICE, dtype=dtype)

        neg_text_input = self.tokenizer(neg_prompt, padding="max_length", max_length=77, return_tensors="pt")
        neg_text_embeddings = self.text_encoder(neg_text_input.input_ids.to(DEVICE))[0]
        dtype = self.unet.dtype
        neg_text_embeddings = neg_text_embeddings.to(device=DEVICE, dtype=dtype)

        # 2) 初始 latent
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.unet.dtype)
        else:
            assert latents.shape == latents_shape

        # 3) 时间表
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        t0 = timesteps[0]

        def _predict_eps(latents, t, text_embeddings,guidance_scale):
            # ---- 关键：把 timestep 对齐到 unet 的设备与 dtype ----
            if not torch.is_tensor(t):
                t = torch.tensor(t, device=latents.device)
            t = t.to(device=latents.device, dtype=self.unet.dtype)
            uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            unconditional_embeddings = unconditional_embeddings.to(dtype)
            
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

            # ---- 关键：把输入和文本特征也对齐 ----
            if abs(guidance_scale) > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            model_inputs = model_inputs.to(dtype=self.unet.dtype)

            te = text_embeddings.to(device=latents.device, dtype=self.unet.dtype)

            eps = self.unet(model_inputs, t, encoder_hidden_states=te).sample

            if abs(guidance_scale) > 1.:
                eps_uc, eps_c = eps.chunk(2, dim=0) #uc无条件 c有条件
                eps = eps_uc + guidance_scale * (eps_c - eps_uc) #CFG
            return eps

        neg_guidance_scale= 0 - guidance_scale
        print(f"neg_guidance={neg_guidance_scale}")
        print(f"guidance={guidance_scale}")
        # 4) 在 t0 上重复 k 次 step→invert （复用现有 step/next_step）
        for _ in range(max(int(k_invert), 0)):
            eps_t0 = _predict_eps(latents, t0, text_embeddings,guidance_scale)
            eps_t0_neg = _predict_eps(latents, t0, neg_text_embeddings,guidance_scale)
            #eps_t0_neg = _predict_eps(latents, t0, neg_text_embeddings,neg_guidance_scale)
            x_tm1, _ = self.step(eps_t0, t0, latents, eta=eta) # step
            latents, _ = self.next_step(eps_t0_neg, t0, x_tm1, eta=eta)    # next_step

        return latents

# 1. 加载模型和调度器
model_id = "runwayml/stable-diffusion-v1-5"  
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = FreePromptPipeline.from_pretrained(
    model_id,
    scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# 2. 定义提示词
#prompt = "Kali, the goddess of time, stands amidst a backdrop of shadows, her body adorned with intricate tattoos and scars, her lips forming a subtle, wistful smile, as if she has witnessed the cyclical nature of human suffering."
#prompt = "A Medea, her visage and garments stained wi\th the crimson hue of ritual sacrifice, stands amidst a makeshift shrine, her presence surrounded by ancient symbols and utensils strewn about the altar. "
##prompt = "An elegant woman in dim light"
prompt = "Andromeda's sorrow is vividly depicted in an image of her sitting within the confines of a white shower cubicle, her eyes streaming tears as she leans forward, her arms wrapped around her knees, her delicate form a poignant reminder of the world's judgment."
neg_prompt = "nuded"#, revealing, sexy
k_invert = 2

# 3. 生成图像
# image = pipe(
#     prompt=prompt,
#     neg_prompt=neg_prompt,
#     height=512, width=512,
#     num_inference_steps=20,
#     guidance_scale=7.5,
#     k_invert=k_invert
# )

image = pipe(
    prompt=prompt,
    neg_prompt=neg_prompt,
    num_inference_steps=20,
    guidance_scale=10,
    k_invert=k_invert
)

# 4. 保存结果

#拆分 prompt，提取前 3 个单词
prompt_words = prompt.split()[:3] 
# 将单词用下划线连接起来，并转换为小写以保证路径规范
prompt_prefix = "_".join(word.lower().strip() for word in prompt_words) 

# 如果 prompt 太短，比如少于 3 个单词，就用全部单词
if not prompt_prefix:
    prompt_prefix = "default_output" # 兜底名称

# 定义路径变量
neg_prompt_str = neg_prompt.replace(" ", "_").strip() # 避免空格，确保路径安全
k_invert_str = str(k_invert)
# 构建完整的保存目录 目录结构: ./output/nake/2/
SAVE_DIR = os.path.join("output", prompt_prefix, neg_prompt_str, k_invert_str )
# 检查并创建目录 (如果不存在则创建，parents=True 允许创建多级目录)
os.makedirs(SAVE_DIR, exist_ok=True) 

timestamp = time.time()
timestamp_str = str(timestamp).replace('.', '_') 

# 文件名格式: output_seed_safe_TIMESTAMP_k_neg.png
file_name = f"{prompt_prefix}_{timestamp_str}.png"
full_path = os.path.join(SAVE_DIR, file_name)

# 保存结果
if isinstance(image, torch.Tensor):
    image = image[0].permute(1, 2, 0).detach().cpu().numpy()
    image = Image.fromarray((image * 255).astype("uint8"))
#使用完整路径保存
image.save(full_path)
print(f"图片已保存到: {full_path}")
