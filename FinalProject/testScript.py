import math
import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
import huggingface_hub
import streamlit as st
import random

# For video display:
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging


torch.manual_seed(1)
# if not Path(huggingface_hub.constants.HF_TOKEN_PATH).exists(): huggingface_hub.notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

col1, col2, col3, col4 = st.columns(4)
columns = [col1, col2, col3, col4]

if 'WEIGHTS' not in st.session_state:
    st.session_state['WEIGHTS'] = [1.0, 1.0, 1.0, 1.0]

@st.cache_resource
def loadModel():
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae")

    # Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14")

    # The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet")

    # The noise scheduler
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    # To the GPU we go!
    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device)
    return vae, tokenizer, text_encoder, unet, scheduler


# @st.cache_data
def createLatents(seed, batch_size, height, width, _scheduler, _unet):
    generator = torch.manual_seed(seed)
    latents = torch.randn(
        (batch_size, _unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]
    # latents = latents * _scheduler.init_noise_sigma
    return latents


def softmax(x):
    sum = 0
    for i in range(0, 4):
        x[i] = math.pow(2, x[i])
        sum += x[i]
    for i in range(0, 4):
        x[i] /= sum
    return x

def updateWeights(x):
    st.session_state['WEIGHTS'] = st.session_state['LISTS'][x]

def generateImage(userPrompt, weights):
    vae, tokenizer, text_encoder, unet, scheduler = loadModel()

    prompt = userPrompt
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 30            # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance 7.5
    batch_size = 1

    # Prep text
    text_input = tokenizer(prompt, padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    randomNum = random.random() / 5 + 0.20
    latentsArray = [createLatents(math.pow(2,5 + i), batch_size, height, width, scheduler, unet) for i in range(4)]
    
    newWeights = torch.tensor([weights[0]+randomNum, weights[1], weights[2], weights[3]])
    newWeights /= torch.linalg.norm(newWeights)

    newWeights1 = torch.tensor([weights[0], weights[1]+randomNum, weights[2], weights[3]])
    newWeights1 /= torch.linalg.norm(newWeights1)

    newWeights2 = torch.tensor([weights[0], weights[1], weights[2]+randomNum, weights[3]])
    newWeights2 /= torch.linalg.norm(newWeights2)

    newWeights3 = torch.tensor([weights[0], weights[1], weights[2], weights[3]+randomNum])
    newWeights3 /= torch.linalg.norm(newWeights3)

    newWeightsArray = [newWeights, newWeights1, newWeights2, newWeights3]

    newWeightsAsList = [weights[0]+randomNum, weights[1], weights[2], weights[3]]
    newWeightsAsList1 = [weights[0], weights[1]+randomNum, weights[2], weights[3]]
    newWeightsAsList2 = [weights[0], weights[1], weights[2]+randomNum, weights[3]]
    newWeightsAsList3 = [weights[0], weights[1], weights[2], weights[3]+randomNum]

    newWeightsLists = [newWeightsAsList, newWeightsAsList1, newWeightsAsList2, newWeightsAsList3]
    st.session_state['LISTS'] = newWeightsLists

    latents = [0, 0, 0, 0]
    for i in range(0,4):
        latents[i] = torch.einsum(
            'a,abchw->bchw',
            newWeightsArray[i],
            torch.stack(latentsArray)
        )
        latents[i] = latents[i].to(torch_device)
        latents[i] *= scheduler.init_noise_sigma
    # for i in range(0, 4):
    #     for j in range(0, 4):
    #         torch.add(latents[i], latentsArray[j], alpha=weightsArray[i][j], out=latents[i])
    #     latents[i] = latents[i].to(torch_device)
    #     latents[i] = latents[i] * scheduler.init_noise_sigma

    # Loop
    with autocast("cuda"):
        for j in range(0, 4):
            for i, t in tqdm(enumerate(scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes

                latent_model_input = torch.cat([latents[j]] * 2)
                sigma = scheduler.sigmas[i]
                # Scale the latents (preconditioning):
                # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, t)
                latent_model_input = latent_model_input.to(torch_device)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t,
                                      encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = scheduler.step(noise_pred, j, latents)["prev_sample"] # Diffusers 0.3 and below

                latents[j] = scheduler.step(
                    noise_pred, t, latents[j]).prev_sample

    # scale and decode the image latents with vae
    for j in range(0, 4):
        latents[j] = 1 / 0.18215 * latents[j]
        with torch.no_grad():
            image = vae.decode(latents[j]).sample

        # Display
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        image = pil_images[0].convert('RGB')
        with columns[j]:
            st.write("Image " + str(j + 1))
            st.image(image)
        del image

    
        
def main():
    userInput = st.text_input("Enter your prompt")
    
    if(st.button("Submit")):
        print(st.session_state['WEIGHTS'])
        generateImage(userInput, st.session_state['WEIGHTS'])
        
    if(st.button("Reset")):
        st.session_state['WEIGHTS'] = [1.0, 1.0, 1.0, 1.0]

    favorite = st.radio("Favorite image", (1, 2, 3, 4), horizontal=True)
    if favorite and st.button("Iterate"):
        updateWeights(favorite - 1)
            
            



main()
