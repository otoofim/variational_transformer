import sys
import os
import glob
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import torch
import torchvision.transforms.functional as TF


class RandomErasing:

    def __call__(self, x):

        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        x = transforms.ToPILImage()(transforms.RandomErasing(p=1.)(x))
        return x

class RandomAffine:

    def __init__(self, angles = (0,360), translate = (2,2), scale = (0,2), shear = (-180,180)):
        self.angle_range = angles
        self.translate = translate
        self.scale_range = scale
        self.shear_range = shear
        self.__new_seed__()

    def __call__(self, x):
        return TF.affine(x, self.angle, self.translate, self.scale, self.shear)

    def __new_seed__(self):
        self.angle = random.uniform(*self.angle_range)
        self.scale = random.uniform(*self.scale_range)
        self.shear = random.uniform(*self.shear_range)


customized_augment_transforms = [
    transforms.ColorJitter(brightness = (0.5,2), contrast = (0.5,2), saturation = (0.5,2), hue = (-0.5,0.5)),
    RandomErasing(),
    transforms.GaussianBlur(9),
    transforms.Grayscale(num_output_channels = 3),
    RandomAffine(angles = (0,360), translate = (0.5,2), scale = (0.5,2), shear = (0,30))
]


def preparing_cityscapes(dataset_add, dest, output_shape:tuple = (256, 256), mode = "train", data_set_mode = "labelIds"):

    count_img = 0
    count_mask = 0
    save_img_add = os.path.join(dest, "images", mode)
    save_mask_add = os.path.join(dest, "masks", mode)
    add = os.path.join(dataset_add, mode, "*")
    final_image_dim = eval(output_shape)

    print(save_img_add)
    if not os.path.exists(save_img_add):
        os.makedirs(save_img_add)


    if not os.path.exists(save_mask_add):
        os.makedirs(save_mask_add)


    for folder in glob.iglob(add):
        for file in glob.iglob(os.path.join(folder,"*")):


            img_add = file
            print(img_add)
            gtFine = img_add.replace("leftImg8bit", "gtFine_{}".format(data_set_mode))
            gtFine = gtFine.replace("gtFine_{}".format(data_set_mode), "gtFine", 1)


            img = Image.open(img_add)
            input_image_shape = np.array(img).shape
            transforms.Resize(final_image_dim)(img).save(os.path.join(save_img_add, "{}.jpg".format(str(count_img).zfill(6))))
            count_img += 1
            img = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(img)

            seg_mask = Image.open(gtFine)
            transforms.Resize(final_image_dim)(seg_mask).save(os.path.join(save_mask_add, "{}.jpg".format(str(count_mask).zfill(6))))
            count_mask += 1
            seg_mask = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(seg_mask)

            for i in range(len(img)):

                transf = random.choice(customized_augment_transforms)
                if hasattr(transf, '__new_seed__'):
                    transf.__new_seed__()

                tmp_img = transforms.Resize(final_image_dim)(img[i])
                tmp_img.save(os.path.join(save_img_add, "{}.jpg".format(str(count_img).zfill(6))))
                count_img += 1
                transf(tmp_img).save(os.path.join(save_img_add, "{}.jpg".format(str(count_img).zfill(6))))
                count_img += 1

                tmp_mask = transforms.Resize(final_image_dim)(seg_mask[i])
                tmp_mask.save(os.path.join(save_mask_add, "{}.jpg".format(str(count_mask).zfill(6))))

                count_mask += 1
                if isinstance(transf, RandomAffine):
                    transf(tmp_mask).save(os.path.join(save_mask_add, "{}.jpg".format(str(count_mask).zfill(6))))

                    count_mask += 1
                else:
                    tmp_mask.save(os.path.join(save_mask_add, "{}.jpg".format(str(count_mask).zfill(6))))
                    count_mask += 1


def main(arg):
    preparing_cityscapes(dataset_add = arg[1], dest = arg[2], output_shape = arg[3], mode = arg[4], data_set_mode = arg[5])

if __name__ == "__main__":
    main(sys.argv)
