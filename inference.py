import argparse
import time
import os
import sys
from pathlib import Path
from PIL import Image

import numpy as np
import torch

sys.path.append(".")

from config import cfg
from train_ctl_model import CTLModel
from utils.reid_metric import get_dist_func

from inference.inference_utils import (
    calculate_centroids,
    create_pid_path_index,
    make_inference_data_loader_from_array,
    run_inference,
)

exctract_func = (
    lambda x: (x).rsplit(".", 1)[0].rsplit("_", 1)[0]
)  ## To extract pid from filename Example: /path/to/dir/product001_04.jpg -> pid = product001
# exctract_func = lambda x: Path(
#     x
# ).parent.name  ## To extract pid from parent directory of an iamge. Example: /path/to/root/001/image_04.jpg -> pid = 001

def get_gallery_set(dataset_dict):
    dataset = []
    pids = []
    for data in dataset_dict:
        dataset.append([data['image'], data['id']])
        pids.append(data['id'])
    return dataset

def load_dummy_gallery(image_dir):
    dataset = []
    image_files = os.listdir(image_dir)
    image_paths= [os.path.join(image_dir, item) for item in image_files]
    for image_file, image_path in zip(image_files, image_paths):
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")

            # extract person id & frame from filename
            id = image_file.split('.')[0]

            dataset.append({
                'id': id,
                'image': img,
                })
    return dataset

def load_dummy_query(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f)
        img.convert("RGB")
    return [[img, 'query']]

def get_result(gallery_roi, out, threshold):
    indices = list(out['indices'])
    gallery_ROI_id = list(out['id'])
    distmat = list(out['distances'])
    matched_id = None

    if distmat[0] < threshold: 
        matched_id = gallery_ROI_id[0]

    for roi in gallery_roi:
        if roi['id'] == matched_id:
            roi['match'] = True
        else:
            roi['match'] = False
        if roi['id'] in gallery_ROI_id:
            idx = gallery_ROI_id.index(roi['id'])
            roi['rank'] = idx + 1
            roi['cosine_distance'] = distmat[idx]
    return gallery_roi 

def inference(model, gallery_set, query_set, top_k=3, normalize_features=True):
    embeddings_gallery, gallery_ROI_id = get_embedding(model, gallery_set)
    embeddings_query, _ = get_embedding(model, query_set)

    ### Create centroids
    # if cfg.MODEL.USE_CENTROIDS:
    # pid_path_index = create_pid_path_index(paths=gallery_ROI_id, func=exctract_func)
    # embeddings_gallery, gallery_ROI_id = calculate_centroids(embeddings_gallery, pid_path_index)

    embeddings_gallery = torch.from_numpy(embeddings_gallery)
    if normalize_features:
        embeddings_gallery = torch.nn.functional.normalize(
            embeddings_gallery, dim=1, p=2
        )
        embeddings_query = torch.nn.functional.normalize(
            torch.from_numpy(embeddings_query), dim=1, p=2
        )
    else:
        embeddings_query = torch.from_numpy(embeddings_query)

    embeddings_gallery = embeddings_gallery.to(device)
    embeddings_query = embeddings_query.to(device)

    t0 = time.time()
    dist_func = get_dist_func(cfg.SOLVER.DISTANCE_FUNC)
    distmat = dist_func(x=embeddings_query, y=embeddings_gallery).cpu().numpy()
    indices = np.argsort(distmat, axis=1)

    t = time.time() - t0
    print("Similarity creation time:", t)

    # ### Constrain the results to only topk most similar ids
    # indices = indices[:, : top_k] if top_k else indices
    print(gallery_ROI_id)
    print(indices)

    out = {
        "indices": indices[0, :],
        "id": gallery_ROI_id[indices[0, :]],
        "distances": distmat[0, indices[0, :]],
    }

    result = get_result(gallery_roi, out, cfg.DISTANCE_THRESHOLD)
    return result

    from pprint import pprint
    pprint(result)
    pprint(out)
    return result

def get_embedding(model, dataset):

    val_loader = make_inference_data_loader_from_array(cfg, dataset)
    if len(val_loader) == 0:
        raise RuntimeError("Lenght of dataloader = 0")

    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False

    ### Inference
    t0 = time.time()
    embeddings, paths = run_inference(
        model, val_loader, cfg, print_freq=10, use_cuda=use_cuda
    )
    t = time.time() - t0
    print("Embedding creation time:", t)
    return embeddings, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create embeddings for images that will serve as the database (gallery)"
    )
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--images-in-subfolders",
        help="if images are stored in the subfloders use this flag. If images are directly under DATASETS.ROOT_DIR path do not use it.",
        action="store_true",
    )
    # parser.add_argument(
    #     "--print_freq",
    #     help="number of batches the logging message is printed",
    #     type=int,
    #     default=10,
    # )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--gallery_data",
        help="path to root where previously prepared embeddings and paths were saved",
        type=str,
    )
    # parser.add_argument(
    #     "--topk",
    #     help="number of top k similar ids to return per query. If set to 0 all ids per query will be returned",
    #     type=int,
    #     default=100,
    # )
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    ### Build model
    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)
    device = torch.device("cuda") if cfg.GPU_IDS else torch.device("cpu")

    gallery_roi = load_dummy_gallery(cfg.DATASETS.GALLERY_DIR)
    query_set = load_dummy_query(cfg.DATASETS.QUERY_PATH)

    gallery_set = get_gallery_set(gallery_roi)
    print(gallery_set)

    result = inference(model, gallery_set, query_set)

