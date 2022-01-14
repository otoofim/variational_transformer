import torchvision.models as models
from torchvision import transforms
from dataloader_cityscapes import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import wandb
import os
import warnings
from statistics import mean
from Variational_transformer import *
from torch.distributions import Normal, Independent, kl, MultivariateNormal
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def train(**kwargs):


    hyperparameter_defaults = {
        "run": kwargs["run_name"],
        "hyper_params": kwargs,
    }

    base_add = os.getcwd()


    if kwargs['continue_tra']:
        wandb.init(config = hyperparameter_defaults, project = kwargs["project_name"], entity = 'moh1371',
                    name = hyperparameter_defaults['run'], resume = "must", id = kwargs["wandb_id"])
        print("wandb resumed...")
    else:
        wandb.init(config = hyperparameter_defaults, project = kwargs["project_name"], entity = 'moh1371',
                    name = hyperparameter_defaults['run'], resume = "allow")


    val_every = 1
    img_w = kwargs["input_img_dim"][0]
    img_h = kwargs["input_img_dim"][1]


    preprocess_in = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        transforms.Resize(kwargs["input_img_dim"])
    ])

    preprocess_ou = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(kwargs["input_img_dim"])
    ])

    tr_loader = CityscapesLoader(dataset_path = kwargs["data_path"], transform_in = preprocess_in, transform_ou = preprocess_ou, mode = 'train')
    train_loader = DataLoader(dataset = tr_loader, batch_size = kwargs["batch_size"], shuffle = True, drop_last = True)
    kwargs["num_cat"] = tr_loader.get_num_classes()

    val_loader = CityscapesLoader(dataset_path = kwargs["data_path"], transform_in = preprocess_in, transform_ou = preprocess_ou, mode = 'val')
    val_loader = DataLoader(dataset = val_loader, batch_size = kwargs["batch_size"], shuffle = True, drop_last = True)


    
    kwargs["posterior_input_channels"] = tr_loader[0]['label'].shape[0] + tr_loader[0]['image'].shape[0]
    kwargs["prior_input_channels"] = tr_loader[0]['image'].shape[0]
    

#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#         print("Running on the GPU")
#     else:
#         device = torch.device("cpu")
#         print("Running on the CPU")

    if kwargs['device'] == "cpu":
        device = torch.device("cpu")
    elif kwargs['device'] == "gpu":
        device = torch.device("cuda:0")
        
    


    model = VariationalTransformer(**kwargs)
    if kwargs["continue_tra"]:
        model.load_state_dict(torch.load(kwargs["model_add"])['model_state_dict'])
        print("model state dict loaded...")

    model = model.to(device)



    optimizer = torch.optim.AdamW(model.parameters(), lr =  kwargs["learning_rate"], weight_decay = kwargs["momentum"])
    criterion = nn.BCEWithLogitsLoss(size_average = False, reduce = False, reduction = None)


    if kwargs["continue_tra"]:
        optimizer.load_state_dict(torch.load(kwargs["model_add"])['optimizer_state_dict'])
        print("optimizer state dict loaded...")


    tr_loss = {"elbo":0., "reconstruct":0., "kl":0.}
    val_loss = 0.
    best_val = 1e10
    beta = 1.0
    wandb.watch(model)

    start_epoch = 0
    end_epoch = kwargs["epochs"]

    if kwargs["continue_tra"]:
        start_epoch = torch.load(kwargs["model_add"])['epoch'] + 1
        end_epoch = torch.load(kwargs["model_add"])['epoch'] + 1 + int(wandb.config.epochs)


    with tqdm(range(start_epoch, end_epoch), unit="epoch", leave = True, position = 0) as epobar:
        for epoch in epobar:
                epobar.set_description("Epoch {}".format(epoch + 1))
                epobar.set_postfix(ordered_dict = {'tr_loss':tr_loss, 'val_loss':val_loss})

                tr_loss = {"elbo":0., "reconstruct":0., "kl":0.}
                out = None
                images = None
                labels = None

                with tqdm(train_loader, unit="batch", leave = False) as batbar:
                    for i, batch in enumerate(batbar):


                        batbar.set_description("Batch {}".format(i + 1))
                        optimizer.zero_grad()

                        #forward pass
                        prior_latent_space, posterior_latent_space, reconstruct_posterior = model.forward(batch['image'].to(device), batch['label'].to(device))
                        
                        #for wandb logging
                        output = reconstruct_posterior[-5:]
                        
                        #calculating lower bound loss function
                        kl_loss = torch.mean(kl.kl_divergence(posterior_latent_space, prior_latent_space))
                        reconstruction_loss = criterion(input = reconstruct_posterior, target = batch['label'])
                        reconstruction_loss = torch.mean(reconstruction_loss)
                        elbo = 1.0 * (reconstruction_loss + (beta * kl_loss))
                        #reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.decoder_emb) + l2_regularisation(model.transformer)
                        #loss = elbo + (kwargs["momentum"] * reg_loss)
                        
                        #back propagation
                        loss = elbo
                        loss.backward()
                        optimizer.step()


                        tr_loss["elbo"] += loss.item()
                        tr_loss["reconstruct"] += reconstruction_loss.item()
                        tr_loss["kl"] += kl_loss.item()

                        images = batch['image'][-5:]
                        labels = batch['seg'][-5 :]



                org_img = {'input': wandb.Image(images),
                "ground truth": wandb.Image(labels),
                "prediction": wandb.Image(tr_loader.prMask_to_color(output))
                 }
                
                wandb.log(org_img)

                tr_loss["elbo"] /= len(train_loader)
                tr_loss["reconstruct"] /= len(train_loader)
                tr_loss["kl"] /= len(train_loader)
                
                wandb.log({"elbo": tr_loss["elbo"], "epoch": epoch + 1})
                wandb.log({"reconstruct": tr_loss["reconstruct"], "epoch": epoch + 1})
                wandb.log({"kl": tr_loss["kl"], "epoch": epoch + 1})


                if ((epoch+1) % val_every == 0):
                    with tqdm(val_loader, unit="batch", leave = False) as valbar:
                        with torch.no_grad():
                            val_loss = 0.0
                            for i, batch in enumerate(valbar):
                                output = torch.zeros(kwargs["batch_size"], tr_loader[0]['label'].shape[0], img_w, img_h)
                                images = batch['image'][-5:]
                                labels = batch['seg'][-5 :]

                                valbar.set_description("Val_batch {}".format(i + 1))
                                optimizer.zero_grad()

                                reconstruction_loss = []
                                for reconstruct_prior in model.inference(batch['image'].to(device)):
                                    loc_loss = torch.mean(criterion(input = reconstruct_prior, target = batch['label'])).item()
                                    reconstruction_loss.append(loc_loss)
                                    output += reconstruct_prior
                                val_loss += mean(reconstruction_loss)


                        val_loss /= len(val_loader)
                        wandb.log({"val_loss": val_loss,
                                   "epoch": epoch + 1,
                                   "input_val": wandb.Image(images),
                                   "ground truth val": wandb.Image(labels),
                                   "prediction val": wandb.Image(tr_loader.prMask_to_color(output/kwargs["num_samples"])[-5:])
                                  })
                        
                        
                        
                        if val_loss < best_val:

                            newpath = os.path.join(base_add, "checkpoints", hyperparameter_defaults['run'])

                            if not os.path.exists(os.path.join(base_add, "checkpoints")):
                                os.makedirs(os.path.join(base_add, "checkpoints"))

                            if not os.path.exists(newpath):
                                os.makedirs(newpath)

                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'tr_loss': tr_loss,
                                'val_loss': val_loss,
                                'hyper_params': kwargs,
                                }, os.path.join(newpath, "best.pth"))
