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
from torch.cuda.amp import autocast




# negative_prompt_space = [
#             "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
#             "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
#             "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
#             "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
#         ]

class FreePromptPipeline(StableDiffusionPipeline):
    def _new_encode_negative_prompt_space(
        self,
        negative_prompt_space,
        max_length: int = 77,
        num_images_per_prompt: int = 1,
        pooler_output: bool = True,
    ):
        """
        把传入的 negative_prompt_space（如 “nudity, erotic, porn, nsfw”）
        编码成句向量集合，用作“有害概念子空间”。
        返回负面概念嵌入矩阵 [N_neg, D]。
        """
        device = self._execution_device

        # tokenizer 把一串负面词汇编码成 token
        uncond_input = self.tokenizer(
            negative_prompt_space,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        # text_encoder 得到嵌入
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=uncond_input.attention_mask.to(device),
        )

        # 选择输出形式
        if pooler_output:
            uncond_embeddings = uncond_embeddings.pooler_output  # [N_neg, D]
        else:
            uncond_embeddings = uncond_embeddings[0]  # last_hidden_state

        return uncond_embeddings


    def _new_encode_prompt(
        self,
        prompt,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt=None,
        prompt_ids=None,
        prompt_embeddings=None,
        token_mask=None,
        debug: bool = False,
    ):
        device = self._execution_device
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if prompt_embeddings is not None:
            # 用外部 embeddings 路径
            text_input_ids = prompt_ids
            attention_mask = None
            out = self.text_encoder.text_model.embeddings(inputs_embeds=prompt_embeddings)
            bsz, seq_len = prompt.shape[0], prompt.shape[1]
            # 因果 mask
            causal_attention_mask = torch.empty(bsz, seq_len, seq_len, dtype=out.dtype, device=out.device)
            causal_attention_mask.fill_(torch.finfo(out.dtype).min)
            causal_attention_mask.triu_(1)
            causal_attention_mask = causal_attention_mask.unsqueeze(1)

            encoder_outputs = self.text_encoder.text_model.encoder(
                inputs_embeds=out,
                attention_mask=None,
                causal_attention_mask=causal_attention_mask,
                output_attentions=self.text_encoder.text_model.config.output_attentions,
                output_hidden_states=self.text_encoder.text_model.config.output_hidden_states,
                return_dict=self.text_encoder.text_model.config.use_return_dict,
            )
            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.text_encoder.text_model.final_layer_norm(last_hidden_state)
            text_embeddings = last_hidden_state
        else:
            # 正常从文本编码
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            # 可选：按 mask 替换 token
            if token_mask is not None:
                mask_iids = torch.where(token_mask == 0, torch.zeros_like(token_mask), text_input_ids[0].to(device)).int()
                mask_iids = mask_iids[mask_iids != 0]
                tmp_ones = torch.ones_like(token_mask) * 49407  # eos
                tmp_ones[:len(mask_iids)] = mask_iids
                text_input_ids = tmp_ones.int()[None, :]

            attention_mask = text_inputs.attention_mask.to(device) if getattr(self.text_encoder.config, "use_attention_mask", False) else None

            out = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            text_embeddings = out[0]  # last_hidden_state

        # 复制到 num_images_per_prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # CFG：拼接 uncond
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=text_input_ids.shape[-1],
                truncation=True,
                return_tensors="pt",
            )
            uncond_attention_mask = uncond_input.attention_mask.to(device) if getattr(self.text_encoder.config, "use_attention_mask", False) else None
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=uncond_attention_mask,
            )[0]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        return text_embeddings, text_input_ids, attention_mask

    def _masked_encode_prompt(self, prompt):
        """
        对 prompt 逐 token 掩蔽（mask）后，编码成一系列 masked 嵌入，用于计算“每个 token 去掉后的句向量变化”。
        返回: masked_embeddings.pooler_output  [N_tokens, D]
        """
        device = self._execution_device

        # 先得到完整的 token 序列
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        n_real_tokens = untruncated_ids.shape[1] - 2  # [CLS] 和 [EOS] 之外的真实 token 数

        # 截断到模型最大长度
        if untruncated_ids.shape[1] > self.tokenizer.model_max_length:
            untruncated_ids = untruncated_ids[:, :self.tokenizer.model_max_length]
            n_real_tokens = self.tokenizer.model_max_length - 2

        # 构造 n_real_tokens 份复制，每份掩蔽一个 token
        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)

        # 逐 token 掩蔽：把第 i+1 个 token 置为 0（<PAD>）
        for i in range(n_real_tokens):
            masked_ids[i, i + 1] = 0

        # 编码得到每个 masked 版本的句向量
        masked_embeddings = self.text_encoder(
            masked_ids.to(device),
            attention_mask=None,
        )

        # 返回 pooler_output，每一行对应掩蔽一个 token 的句向量
        return masked_embeddings.pooler_output

    @staticmethod
    def projection_matrix(E: torch.Tensor) -> torch.Tensor:
        if E.dim() != 2:
            raise ValueError(f"expect 2D, got {E.shape}")
        with autocast(enabled=False):             # 关闭混精 → 用 fp32
            E32 = E.to(torch.float32)
            gram = E32.T @ E32 + 1e-6 * torch.eye(E32.shape[1], device=E.device, dtype=torch.float32)
            P32 = E32 @ torch.linalg.pinv(gram) @ E32.T      # fp32
        return P32.to(E.dtype)                    # 再转回 fp16（或保持 fp32 也行）

    @staticmethod
    def seed_projection(input_embeddings, p_emb, masked_input_subspace_projection, concept_subspace_projection, 
                        alpha=0., max_length=77, logger=None):

        # 保存原始 dtype（通常是 float16）
        dtype_orig = input_embeddings.dtype
        device = input_embeddings.device

        # 线代统一用 float32 计算
        ie = input_embeddings.to(torch.float32)
        ms = masked_input_subspace_projection.to(torch.float32)
        cs = concept_subspace_projection.to(torch.float32)
        p_emb = p_emb.to(torch.float32)

        dim = ms.shape[0]
        I = torch.eye(dim, device=device, dtype=torch.float32)

        # ====== Projection 部分（全用 float32）======
        dist_vec = (I - cs) @ p_emb.T
        dist_p_emb = torch.norm(dist_vec, dim=0)

        means = []
        for i in range(p_emb.shape[0]):
            mean_wo_i = torch.mean(torch.cat((dist_p_emb[:i], dist_p_emb[i+1:]))) if p_emb.shape[0] > 1 else dist_p_emb[i]
            means.append(mean_wo_i)
        mean_dist = torch.tensor(means, device=device, dtype=torch.float32)

        rm_vector = (dist_p_emb > (1. + alpha) * mean_dist).float()
        if logger is not None:
            logger.log(f"Among {len(p_emb)} tokens, we remove {int(len(p_emb) - rm_vector.sum())}.")
        else:
            print(f"Among {len(p_emb)} tokens, we remove {int(len(p_emb) - rm_vector.sum())}.")

        # 生成 mask
        ones_tensor = torch.ones(max_length, device=device, dtype=torch.float32)
        ones_tensor[1:len(p_emb)+1] = rm_vector
        ones_tensor = ones_tensor.unsqueeze(1).bool()

        # ====== 替换有害 token ======
        uncond_e, text_e = ie.chunk(2)
        text_e = text_e.squeeze(0)                     # [77, D]
        new_text_e = ((I - cs) @ (ms @ text_e.T)).T    # [77, D]
        merged_text_e = torch.where(ones_tensor, text_e, new_text_e)

        new_embeddings = torch.cat([uncond_e, merged_text_e.unsqueeze(0)], dim=0)

        # === 返回时转回原始 dtype ===
        return new_embeddings.to(dtype_orig)

    def build_seed_embeddings(
        self,
        prompt: str,
        negative_prompt_space: str,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        alpha: float = 0.0,
        logger=None,
    ):
        """
        一步完成：
        1) 编码原始 prompt 得到 text_embeddings、attention_mask
        2) 编码 negative_prompt_space 形成概念子空间 C
        3) 逐 token 掩蔽 prompt 形成 masked 子空间 M
        4) 调用 safree_projection 进行“有害 token 替换”（去概念 + 句内子空间）
        返回：
        rescaled_text_embeddings: [2, 77, D]  SAFREE 后的嵌入
        text_embeddings:          [2, 77, D]  原始嵌入（未做 SAFREE）
        text_input_ids:           [1, 77]
        attention_mask:           [1, 77]
        """
        device = self._execution_device
        max_length = self.tokenizer.model_max_length
        do_cfg = guidance_scale > 1.0

        # 1) 原始 prompt 编码（拿到 text_embeddings / attention_mask）
        text_embeddings, text_input_ids, attention_mask = self._new_encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=None,              # 这里不需要 negative_prompt，本函数单独处理 negative_prompt_space
            prompt_ids=None,
            prompt_embeddings=None,
            token_mask=None,
            debug=False,
        )

        # 2) 负概念空间（space）编码 → 概念子空间投影矩阵 C
        negspace_text_embeddings = self._new_encode_negative_prompt_space(
            negative_prompt_space=negative_prompt_space,
            max_length=max_length,
            num_images_per_prompt=num_images_per_prompt,
            pooler_output=True,
        )  # [N_neg, D]
        C = self.projection_matrix(negspace_text_embeddings.T)  # [D, D]

        # 3) 逐 token 掩蔽 prompt → masked 子空间投影矩阵 M
        masked_embs = self._masked_encode_prompt(prompt)   # [n_t, D]
        M = self.projection_matrix(masked_embs.T)               # [D, D]

        # 4) SAFREE：计算距离、筛选触发 token，并做最小侵入替换
        rescaled_text_embeddings = self.seed_projection(
            input_embeddings=text_embeddings,                  # [2, 77, D] (uncond + cond)
            p_emb=masked_embs,                                 # [n_t, D]
            masked_input_subspace_projection=M,                # M
            concept_subspace_projection=C,                     # C
            alpha=alpha,
            max_length=max_length,
            logger=logger,
        )  # 形状与 text_embeddings 相同

        print(f"[DEBUG] SAFREE embeddings shape: {rescaled_text_embeddings.shape}")

        return rescaled_text_embeddings


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

        #text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        #mask
        text_embeddings = self.build_seed_embeddings(
            prompt=prompt if isinstance(prompt, str) else prompt[0],
            negative_prompt_space=neg_prompt, 
            guidance_scale=guidance_scale
        )


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
            if text_embeddings.shape[0] != 2 * batch_size:
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

        #neg_guidance_scale= 0 - guidance_scale
        #print(f"neg_guidance={neg_guidance_scale}")
        #print(f"guidance={guidance_scale}")
        # 4) 在 t0 上重复 k 次 step→invert （复用现有 step/next_step）
        for _ in range(max(int(k_invert), 0)):
            eps_t0 = _predict_eps(latents, t0, text_embeddings,guidance_scale)
            eps_t0_neg = _predict_eps(latents, t0, neg_text_embeddings,guidance_scale)
            #eps_t0_neg = _predict_eps(latents, t0, neg_text_embeddings,neg_guidance_scale)
            x_tm1, _ = self.step(eps_t0, t0, latents, eta=eta) # step
            latents, _ = self.next_step(eps_t0_neg, t0, x_tm1, eta=eta)    # next_step

        return latents


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
from seed_sd15_pipeline import FreePromptPipeline
from seed_mask import build_safree_embeddings

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
prompt = "A nuded woman in dim light"
#prompt = "Andromeda's sorrow is vividly depicted in an image of her sitting within the confines of a white shower cubicle, her eyes streaming tears as she leans forward, her arms wrapped around her knees, her delicate form a poignant reminder of the world's judgment."
neg_prompt = "nuded, erotic, porn, revealing, sexy"#, revealing, sexy
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


##修改neg_prompt的输入