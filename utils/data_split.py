import glob
import os
import json
import random
import SimpleITK as sitk
import csv
import nibabel as nib
import numpy as np


def to_csv(path, infos):
    field_names = list(infos[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(infos)


def get_path_list(dir_path):
    total_path = {}
    for root_dir in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, root_dir)):
            continue
        path = []
        for patient_dir in os.listdir(os.path.join(dir_path, root_dir)):
            if not os.path.isdir(os.path.join(dir_path, root_dir, patient_dir)):
                continue
            files = os.listdir(os.path.join(dir_path, root_dir, patient_dir))
            path_dict = {}

            for m in files:
                path_dict.update(
                    {f"moving_{m.split('_')[-1].split('.')[0]}": os.path.abspath(
                        os.path.join(dir_path, root_dir, patient_dir, m))}
                )
            path.append(path_dict)
        if 'train' in root_dir.split('_')[-1].lower():
            total_path.update({'train': path})
        if 'val' in root_dir.split('_')[-1].lower():
            total_path.update({'val': path})
        if 'test' in root_dir.split('_')[-1].lower():
            total_path.update({'test': path})
    return total_path


def get_brats_path(dir_path, c):
    total_path = []
    for patient_dir in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, patient_dir)):
            continue
        files = os.listdir(os.path.join(dir_path, patient_dir))
        path_dict = {}

        for m in files:
            path_dict.update(
                {f"moving_{m.split('_')[-1].split('.')[0]}": os.path.abspath(
                    os.path.join(dir_path, patient_dir, m))}
            )
        total_path.append(path_dict)
    return total_path


def get_brats_reg_withlandmark_path(dir_path, c):
    total_path = []
    for patient_dir in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, patient_dir)):
            continue
        files = os.listdir(os.path.join(dir_path, patient_dir))
        path_dict = {}

        for m in files:
            if 'lta' in m or '_landmarks_' in m:
                continue

            if 'pre' in m:
                path_dict.update(
                    {f"moving_{m.split('.')[0].split('_')[1]}": os.path.abspath(os.path.join(dir_path, patient_dir, m))}
                )

            if 'post' in m:
                path_dict.update(
                    {f"fixed_{m.split('.')[0].split('_')[1]}": os.path.abspath(os.path.join(dir_path, patient_dir, m))}
                )
        total_path.append(path_dict)
    return total_path


def get_oasis_path(dir_path, c='t1'):
    total_path = []
    for patient_dir in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, patient_dir)):
            continue
        files = os.listdir(os.path.join(dir_path, patient_dir))
        path_dict = {}

        for m in files:
            if m.split('.')[-1] != 'gz':
                continue
            path_dict.update(
                {f"moving_{m.split('.')[0].replace('brain', c).replace('aseg', 'seg')}": os.path.abspath(
                    os.path.join(dir_path, patient_dir, m))}
            )
        total_path.append(path_dict)
    return total_path


def get_stroke_path(dir_path, c='t1'):
    total_path = []
    for patient_dir in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, patient_dir)):
            continue
        files = os.listdir(os.path.join(dir_path, patient_dir))
        path_dict = {}

        for m in files:
            if m.split('.')[-1] != 'gz':
                continue
            if 'brain' in m:
                path_dict.update(
                    {f"moving_t1": os.path.abspath(os.path.join(dir_path, patient_dir, m))}
                )
            if 'seg' in m:
                path_dict.update(
                    {f"moving_seg": os.path.abspath(os.path.join(dir_path, patient_dir, m))}
                )
        total_path.append(path_dict)
    return total_path


def get_pseudo_path(dir_path, c='t1'):
    total_path = []
    for patient_dir in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, patient_dir)):
            continue
        files = os.listdir(os.path.join(dir_path, patient_dir))
        path_dict = {}

        for m in files:
            if m.split('.')[-1] != 'gz':
                continue
            if 'oasis_brain' in m:
                path_dict.update(
                    {"normal": os.path.abspath(
                        os.path.join(dir_path, patient_dir, m))}
                )
            else:
                path_dict.update(
                    {f"moving_{m.split('.')[0].replace('brain', c)}": os.path.abspath(
                        os.path.join(dir_path, patient_dir, m))}
                )
        total_path.append(path_dict)
    return total_path


def to_json(dir_path='./original_data/', c='t1', fold_list=None, data_list=None, fold=0, test_list=None):
    train_list, val_list = ([], []) if data_list is None else data_list
    if fold_list is not None:
        for i in range(5):
            if i == fold:
                val_list.extend(fold_list[i])
            else:
                train_list.extend(fold_list[i])

    file = open(os.path.join(dir_path, f'data_fold_{fold}.json'), 'w')
    jsons = dict()
    jsons['description'] = 'Penn Challenge'
    jsons["licence"] = 'Penn'
    jsons['modality'] = {'0': 'MRI'}
    jsons.update({"name": "OASIS",
                  "numTest": len(val_list),
                  "numTraining": len(train_list),
                  "reference": "UESTC University",
                  "release": "1.0 06/08/2022",
                  "tensorImageSize": "3D", })
    if test_list is None:
        jsons.update({
            'training': sorted(train_list, key=lambda x: x[f'moving_{c}']),
            'validation': sorted(val_list, key=lambda x: x[f'moving_{c}'])
        })
    else:
        jsons.update({
            'training': sorted(train_list, key=lambda x: x[f'moving_{c}']),
            'validation': sorted(val_list, key=lambda x: x[f'moving_{c}']),
            'test': sorted(test_list, key=lambda x: x[f'moving_{c}'])
        })
    json.dump(jsons, file)
    file.close()


def read_json(path):
    f = open(path, 'r')
    str_json = json.load(f)
    return str_json


def read_data(path):
    img = sitk.ReadImage(path)
    return img


def get_info(img: sitk.Image):
    info = dict()
    info['Size'] = img.GetSize()
    info['Origin'] = img.GetOrigin()
    info['Range'] = (
        sitk.GetArrayFromImage(img).min(), sitk.GetArrayFromImage(img).mean(), sitk.GetArrayFromImage(img).max())
    info['Direction'] = img.GetDirection()
    info['Spacing'] = img.GetSpacing()
    for k in img.GetMetaDataKeys():
        info[k] = img.GetMetaData(k)
    return info


def dirpath_replace(json_dir_path, new_dir):
    for json_p in glob.glob(os.path.join(json_dir_path, '*')):
        str_json = read_json(json_p)
        train_list, valid_list = str_json['training'], str_json['validation']
        for t in train_list:
            for k, v in t.items():
                t[k] = os.path.join(new_dir, *v.split('/')[-2:])

        for val in valid_list:
            for k, v in val.items():
                val[k] = os.path.join(new_dir, *v.split('/')[-2:])


def get_stroke_threshold_path(threshold=0.01, num=None):
    def fn(dir_path, c='t1'):
        data_list = open('/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokePseudo.txt').read().splitlines()
        data_list = list(map(lambda x: [x.split(' ')[0], float(x.split(' ')[1])], data_list))
        if num is None:
            for i in range(len(data_list) - 1, -1, -1):
                if data_list[i][1] <= threshold:
                    del data_list[i]
        else:
            data_list = data_list[:num]
        data_list = list(map(lambda x: x[0], data_list))

        total_path = []
        for patient_dir in os.listdir(dir_path):
            if not os.path.isdir(os.path.join(dir_path, patient_dir)) or patient_dir not in data_list:
                continue
            files = os.listdir(os.path.join(dir_path, patient_dir))
            path_dict = {}

            for m in files:
                if m.split('.')[-1] != 'gz':
                    continue
                if 'oasis_brain' in m:
                    path_dict.update(
                        {"atlas": os.path.abspath(
                            os.path.join(dir_path, patient_dir, m))}
                    )
                else:
                    path_dict.update(
                        {f"moving_{m.split('.')[0].replace('brain', c)}": os.path.abspath(
                            os.path.join(dir_path, patient_dir, m))}
                    )
            total_path.append(path_dict)
        return total_path

    return fn


def calculate_lesion_size(dir_path, file_path_list, fresh=False):
    if 'moving_seg' not in file_path_list[0].keys() or 'aseg' in file_path_list[0]['moving_seg']:
        return None

    if not os.path.exists(os.path.join(dir_path, 'volume.txt')) or fresh:
        print_str = []
        for path in file_path_list:
            mask = nib.load(path['moving_seg']).get_fdata()
            mask = (mask > .5).astype(float)
            ratio = mask.sum() / np.prod(mask.shape)
            print_str.append(f'{path["moving_seg"].split("/")[-2]}_{ratio}')
        print_str = sorted(print_str, key=lambda x: float(x.split('_')[-1]), reverse=True)
        txt = open(os.path.join(dir_path, 'volume.txt'), 'w')
        txt.write('\n'.join(print_str))
        txt.close()
    else:
        print_str = open(os.path.join(dir_path, 'volume.txt'), 'r').read().splitlines()
    return list(map(lambda x: '_'.join(x.split('_')[:-1]), print_str))


def create_split(save_dir, ratio=(5, 1, 1), dir_path='../data/OASIS', contrast='t1', get_path=None, num=None):
    os.makedirs(save_dir, exist_ok=True)
    file_path = get_path(dir_path, contrast)
    random.shuffle(file_path)

    if num:
        name = calculate_lesion_size(dir_path=dir_path, file_path_list=file_path)
        if name:
            file_path = [i for i in get_path(dir_path, contrast) if i['moving_t1'].split('/')[-2] in name[:num]]
        else:
            file_path = file_path[:num]

    fold_list = []
    fold_len = int(len(file_path) / np.prod(ratio))
    for i in range(5):
        fold_list.append(file_path[i * fold_len:(i + 1) * fold_len])
    for i in range(5):
        to_json(save_dir, c=contrast, fold_list=fold_list, fold=i)


def change_rootdir(path='../data_json'):
    for dataset in os.listdir(path):
        for file_path in os.listdir(os.path.join(path, dataset)):
            file = json.load(open(os.path.join(path, dataset, file_path), 'r'))
            training = file['training']
            validation = file['validation']
            for data_dict in training:
                for k, v in data_dict.items():
                    data_dict[k] = v.replace('/data_smr/liuy/Challenge/PennBraTSReg', '/data_58/liuy')
            for data_dict in validation:
                for k, v in data_dict.items():
                    data_dict[k] = v.replace('/data_smr/liuy/Challenge/PennBraTSReg', '/data_58/liuy')
            os.makedirs(f'../data_json2/{dataset}', exist_ok=True)
            json.dump(file, open(os.path.join('../data_json2', dataset, file_path), 'w'))


if __name__ == '__main__':
    # create_split('../data_json/train_OASISNLin', dir_path='../data/OASISMNI152NLin', get_path=get_oasis_path)
    # create_split('../data_json/train_OASISNLin100', num=100, dir_path='../data/OASISMNI152NLin',
    #              get_path=get_oasis_path)
    #
    # create_split('../data_json/train_BraTSNLin', dir_path='../data/BraTS2020MNI152NLin', get_path=get_brats_path,
    #              num=300)
    # create_split('../data_json/train_BraTSNLin100', num=100, dir_path='../data/BraTS2020MNI152NLin',
    #              get_path=get_brats_path)
    #
    # create_split('../data_json/train_BraTSPseudoNLin', dir_path='../data/BraTSPseudo', get_path=get_pseudo_path)
    # create_split('../data_json/train_BraTSPseudoHistoNLin', dir_path='../data/BraTSPseudoHisto', num=None,
    #              get_path=get_pseudo_path)
    # create_split('../data_json/train_StrokePseudoNLin', dir_path='../data/StrokePseudoHisto', get_path=get_pseudo_path)
    # create_split('../data_json/train_StrokePseudoNLin100', num=100, dir_path='../data/StrokePseudo',
    #              get_path=get_pseudo_path)

    # create_split('../data_json/train_StrokeNLin100', num=100, dir_path='../data/StrokeMNI152NLin',
    #              get_path=get_stroke_path)
    # create_split('../data_json/train_BraTSRegLandmarkNLin',
    #              dir_path='../data/BraTSRegWithLandmarkNLin',
    #              get_path=get_brats_reg_withlandmark_path)

    create_split('../data_json/train_BraTSRegLandmarkRAS',
                 dir_path='../data/BraTSRegWithLandmarkRAS',
                 get_path=get_brats_reg_withlandmark_path)

    # change_rootdir()
