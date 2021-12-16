import torchvision.models as models
from torchvision import transforms
from dataloader import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import wandb
from unet import *
import os
import warnings
from statistics import mean
warnings.filterwarnings("ignore")

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

# def elbo(segm, prior_latent_space, posterior_latent_space, reconstruct_posterior, reconstruct_prior):
#     """
#     Calculate the evidence lower bound of the log-likelihood of P(Y|X)
#     """
#
#     criterion = nn.BCEWithLogitsLoss(size_average = False, reduce = False, reduction = None)
#     kl = torch.mean(kl.kl_divergence(posterior_latent_space, prior_latent_space))
#     reconstruction_loss = criterion(input = reconstruct_posterior, target = segm)
#     reconstruction_loss = torch.sum(reconstruction_loss)
#     #mean_reconstruction_loss = torch.mean(reconstruction_loss)
#
#     return -(reconstruction_loss + (beta * kl))

def preprocessing():

    tt = [transforms.ColorJitter(),
    transforms.RandomCrop(64),
    transforms.RandomRotation((0,360)),
    transforms.GaussianBlur(2),
    transforms.RandomErasing(),
    ]

    preprocess_in = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomChoice(tt)
        transforms.Resize((img_w, img_h))
    ])

    preprocess_ou = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_w, img_h))
    ])


def train(**kwargs):


    hyperparameter_defaults = {
        "batch_size": kwargs["batch_size"],
        "lr": kwargs["lr"],
        "epochs": kwargs["epochs"],
        "momentum": kwargs["momentum"],
        "architecture": kwargs["architecture"],
        "dataset": kwargs["dataset"],
        "run": kwargs["run"]
    }

    base_add = os.getcwd()


    if continue_tra:
        wandb.init(config = hyperparameter_defaults, project = kwargs["project_name"], entity = 'moh1371',
                    name = hyperparameter_defaults['run'], resume = "must", id = kwargs["wandb_id"])
        print("wandb resumed...")
    else:
        wandb.init(config = hyperparameter_defaults, project = kwargs["project_name"], entity = 'moh1371',
                    name = hyperparameter_defaults['run'], resume = "allow")


    val_every = 2
    img_w = kwargs["input_dim"]
    img_h = kwargs["input_dim"]

    input_preproc = preprocessing()
    gt_preproc = preprocessing()


    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")


    model = VariationalTransformer(**kwargs)
    if kwargs["continue_tra"]:
        model.load_state_dict(torch.load(kwargs["model_add"])['model_state_dict'])
        print("model state dict loaded...")

    model = unet.to(device)

    tr_loader = CityscapesLoader(data_path, preprocess_in, preprocess_ou, mode = 'train')
    train_loader = DataLoader(dataset = tr_loader, batch_size = wandb.config.batch_size, shuffle = True)

    val_loader = CityscapesLoader(data_path, preprocess_in, preprocess_ou, mode = 'val')
    val_loader = DataLoader(dataset = val_loader, batch_size = wandb.config.batch_size, shuffle = True)

    #weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters, lr =  kwargs["continue_tra"], weight_decay =  0)
    criterion = nn.BCEWithLogitsLoss(size_average = False, reduce = False, reduction = None)


    if kwargs["continue_tra"]:
        optimizer.load_state_dict(torch.load(kwargs["model_add"])['optimizer_state_dict'])
        print("optimizer state dict loaded...")


    tr_loss = 0.0
    val_loss = 0.0
    best_val = 1e10
    wandb.watch(model)

    start_epoch = 0
    end_epoch = wandb.config.epochs

    if kwargs["continue_tra"]:
        start_epoch = torch.load(kwargs["model_add"])['epoch'] + 1
        end_epoch = torch.load(kwargs["model_add"])['epoch'] + 1 + int(wandb.config.epochs)


    with tqdm(range(start_epoch, end_epoch), unit="epoch", leave = True, position = 0) as epobar:
        for epoch in epobar:
                epobar.set_description("Epoch {}".format(epoch + 1))
                epobar.set_postfix(ordered_dict = {'tr_loss':tr_loss, 'val_loss':val_loss})

                tr_loss = 0.0
                out = None
                images = None
                labels = None

                with tqdm(train_loader, unit="batch", leave = False) as batbar:
                    for i, batch in enumerate(batbar):


                        batbar.set_description("Batch {}".format(i + 1))
                        optimizer.zero_grad()

                        prior_latent_space, posterior_latent_space, reconstruct_posterior = model.forward(batch['image'], batch['label'])

                        #elbo = elbo(batch['label'], prior_latent_space, posterior_latent_space, reconstruct_posterior, reconstruct_prior)

                        kl = torch.mean(kl.kl_divergence(posterior_latent_space, prior_latent_space))
                        reconstruction_loss = criterion(input = reconstruct_posterior, target = batch['label'])
                        reconstruction_loss = torch.sum(reconstruction_loss)
                        #mean_reconstruction_loss = torch.mean(reconstruction_loss)

                        elbo = reconstruction_loss + (beta * kl)
                        reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.decoder_emb) + l2_regularisation(model.transformer)
                        loss = elbo + (kwargs["continue_tra"] * reg_loss)
                        loss.backward()
                        optimizer.step()


                        tr_loss += loss.item()

                        images = batch['image']
                        labels = batch['label']


                org_img = {'input':wandb.Image(batch['image']),
                "ground truth":wandb.Image(batch['label']),
                "prediction":wandb.Image(out)}

                wandb.log(org_img)


                tr_loss /= len(train_loader)
                wandb.log({"tr_loss": tr_loss, "epoch": epoch + 1})


                if ((epoch+1) % val_every == 0):
                    with tqdm(val_loader, unit="batch", leave = False) as valbar:
                        with torch.no_grad():
                            val_loss = 0.0
                            for i, batch in enumerate(valbar):

                                valbar.set_description("Val_batch {}".format(i + 1))
                                optimizer.zero_grad()

                                reconstruction_loss = []
                                for reconstruct_prior in model.inference(batch['image']):
                                    reconstruction_loss.append(criterion(input = reconstruct_prior, target = batch['label']).item())

                                val_loss += mean(reconstruction_loss)


                        val_loss /= len(val_loader)
                        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
                        if val_loss < best_val:

                            newpath = base_add + "/checkpoints/{}".format(hyperparameter_defaults['run'])

                            if not os.path.exists(base_add + "/checkpoints"):
                                os.makedirs(base_add + "/checkpoints")

                            if not os.path.exists(newpath):
                                os.makedirs(newpath)

                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'tr_loss': tr_loss,
                                'val_loss': val_loss,
                                }, newpath + "/best.pth")
