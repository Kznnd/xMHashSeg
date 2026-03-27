import torch

def get_collate(input_dict_list):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    labels=[]
    imgs = []
    img_idxs = []
    pc = []
    locs = []
    feats = []

    for idx, input_dict in enumerate(input_dict_list):
        if 'coords' in input_dict.keys():
            coords = torch.from_numpy(input_dict['coords'])
            batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
            locs.append(torch.cat([coords, batch_idxs], 1))
            feats.append(torch.from_numpy(input_dict['feats']))

        labels.append(torch.from_numpy(input_dict['seg_label']))
        imgs.append(torch.from_numpy(input_dict['img']))
        img_idxs.append(input_dict['img_indices'])
        pc.append(torch.from_numpy(input_dict["pc"]))

    if 'coords' in input_dict.keys():
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        out_dict = {'x': [locs, feats]}
    else:
        out_dict = {}
    labels = torch.cat(labels, 0)
    out_dict['seg_label'] = labels
    out_dict['img'] = torch.stack(imgs)
    out_dict['img_indices'] = img_idxs
    out_dict['pc'] = torch.stack(pc)
    out_dict['num_classes'] = input_dict['num_classes']

    return out_dict
