import copy
import os
from argparse import Namespace

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

from modules import UNet, EMA
from utils import setup_logging, get_data, save_images, plot_images
import numpy as np


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%TM:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.002, image_size=64, device="cpu"):
        self.device = device
        self.image_size = image_size
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.beta_start = beta_start

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timestepss(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images ...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, labels, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            model.train()
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x


def train(config):
    exp_ns = Namespace(**config["experiment"])
    dataset_ns = Namespace(**config["dataset"])
    model_ns = Namespace(**config["model"])
    training_ns = Namespace(**config["training"])

    setup_logging(exp_ns.run_name)
    device = exp_ns.device

    dataloader = get_data(dataset_ns.train_path, training_ns.batch_size, dataset_ns.image_size)
    model = UNet(num_classes=model_ns.num_classes, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), **config["optimizer"])
    mse = nn.MSELoss()
    diffusion = Diffusion(image_size=dataset_ns.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", exp_ns.run_name))
    l = len(dataloader)
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(training_ns.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timestepss(training_ns.batch_size).to(device)
            x_t, noise = diffusion.noise_image(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=training_ns.batch_size, labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=training_ns.batch_size, labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", exp_ns.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", exp_ns.run_name, f"{epoch}_ema.jpg"))
            torch.save(ema_model.state_dict(), os.path.join("models", exp_ns.run_name, f"ckpt_ema.pt"))
            torch.save(model.state_dict(), os.path.join("models", exp_ns.run_name, f"ckpt.pt"))

