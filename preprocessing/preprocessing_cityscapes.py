


def preparing_cityscapes(dataset_add, dest, output_shape:tuple = (256, 256), mode = "train", data_set_mode = "labelIds"):


    count_img = 0
    count_mask = 0
    save_img_add = os.path.join(dest, "images", mode)
    save_mask_add = os.path.join(dest, "masks", mode)
    add = os.path.join(dataset_add, mode, "*")
    final_image_dim = output_shape


    for folder in glob.iglob(add):
        for file in glob.iglob(os.path.join(folder,"*")):


            img_add = file
            print(img_add)
            gtFine = img_add.replace("leftImg8bit", "gtFine_{}".format(data_set_mode))
            gtFine = gtFine.replace("gtFine_{}".format(data_set_mode), "gtFine", 1)


            img = Image.open(img_add)
            input_image_shape = np.array(img).shape
            transforms.Resize(final_image_dim)(img).save(save_img_add + "{}.jpg".format(str(count_img).zfill(6)))
            count_img += 1
            img = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(img)

            seg_mask = Image.open(gtFine)
            transforms.Resize(final_image_dim)(seg_mask).save(save_mask_add + "{}.jpg".format(str(count_mask).zfill(6)))
            count_mask += 1
            seg_mask = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(seg_mask)

            for i in range(len(img)):

                transf = random.choice(customized_augment_transforms)
                if hasattr(transf, '__new_seed__'):
                    transf.__new_seed__()

                tmp_img = transforms.Resize(final_image_dim)(img[i])
                tmp_img.save(save_img_add + "{}.jpg".format(str(count_img).zfill(6)))
                count_img += 1
                transf(tmp_img).save(save_img_add + "{}.jpg".format(str(count_img).zfill(6)))
                count_img += 1

                tmp_mask = transforms.Resize(final_image_dim)(seg_mask[i])
                tmp_mask.save(save_mask_add + "{}.jpg".format(str(count_mask).zfill(6)))

                count_mask += 1
                if isinstance(transf, RandomAffine):
                    transf(tmp_mask).save(save_mask_add + "{}.jpg".format(str(count_mask).zfill(6)))

                    count_mask += 1
                else:
                    tmp_mask.save(save_mask_add + "{}.jpg".format(str(count_mask).zfill(6)))
                    count_mask += 1


def main(arg):
    preparing_cityscapes(dataset_add = arg[0], dest = arg[1], output_shape:tuple = arg[2], mode = arg[3], data_set_mode = [4])

if __name__ == "__main__":
    main(sys.argv)
