import numpy as np
import os
from glob import glob
from lib.config import cfg
from lib.utils.data_utils import load_K_Rt_from_P, read_camera
from lib.utils import base_utils, data_utils
from lib.datasets import enerf_utils
import imp
from os.path import join
import cv2
import random
import tqdm
import trimesh
from termcolor import colored
import imageio
import torch
if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

from scipy import interpolate

from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils.collate import default_convert, default_collate


from lib.utils.data_utils import to_cuda, add_batch
from lib.utils.net_utils import gen_rays_bbox, perf_timer
from lib.config import cfg

import ipdb

from typing import Mapping, Collection

timer = perf_timer(use_ms=True, logf=lambda x: print(colored(x, "green")), disabled=True)


class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()

        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        b, e, s = kwargs['frames']
        self.render_frames = np.arange(b, e)[::s].tolist()
        self.input_h_w = kwargs['input_h_w']
        self.input_ratio = kwargs['input_ratio']

        scene = kwargs['scene']
        self.scene = scene
        self.build_metas(kwargs)

    def build_metas(self, config):
        scene_info = {'ixts': [], 'exts': [], 'Ds': [], 'bbox': {}}
        scene_root = join(self.data_root, self.scene)
        self.scene_root = scene_root
        intri_path = join(scene_root, 'intri.yml')
        extri_path = join(scene_root, 'extri.yml')
        cams = read_camera(intri_path, extri_path)
        self.cams = cams
        cam_ids = sorted([item for item in os.listdir(join(scene_root, 'images')) if item[0] != '.'])
        cam_len = len(cam_ids)
        ixts = np.array([self.cams[cam_id]['K'] for cam_id in cam_ids]).reshape(cam_len, 3, 3).astype(np.float32)
        exts = np.array([self.cams[cam_id]['RT'] for cam_id in cam_ids]).reshape(cam_len, 3, 4).astype(np.float32)
        Ds = np.array([self.cams[cam_id]['dist'] for cam_id in cam_ids]).reshape(cam_len, 5).astype(np.float32)
        exts_ones = np.zeros_like(exts[:, :1, :])
        exts_ones[..., 3] = 1.
        exts = np.concatenate([exts, exts_ones], axis=1)
        scene_info['exts'] = exts
        scene_info['ixts'] = ixts
        scene_info['Ds'] = Ds

        frame_len = len(glob(f'{scene_root}/images/00/*.jpg'))
        b, e, s = config['frames']
        e = e if e != -1 else frame_len
        self.scene_info = scene_info

        for frame_id in tqdm.tqdm(np.arange(frame_len)[b:e:s]):
            corners_3d =  np.array([
                [-5.0, -2.0, 0.0],
                [-5.0, -2.0, 60.0],
                [-5.0, 0.0, 0.0],
                [-5.0, 0.0, 60.0],
                [45.0, -2.0, 0.0],
                [45.0, -2.0, 60.0],
                [45.0, 0.0, 0.0],
                [45.0, 0.0, 60.0],
            ])/1000
            scene_info['bbox'][frame_id] = corners_3d

        points = np.array(trimesh.load(join(scene_root, 'background.ply')).vertices)
        scene_info['bbox_dynamic'] = points
        self.bkgd_near_far = []
        self.ixts=[]
        for view_id in range(len(cam_ids)):
            img, ext, ixt = self.read_data(view_id, 1)
            h, w = img.shape[:2]
            points_ = points @ ext[:3, :3].T + ext[:3, 3].T
            uv = points_ @ ixt.T
            uv[:, :2] = uv[:, :2] / uv[:, 2:]
            mask = np.logical_or(uv[..., 0] < 0, uv[..., 1] < 0)
            mask = np.logical_or(mask, uv[..., 0] > w - 1)
            mask = np.logical_or(mask, uv[..., 1] > h - 1)
            uv = uv[mask == False]
            near_far = np.array([uv[:, 2].min(), uv[:, 2].max()])
            self.ixts.append(ixt)
            self.bkgd_near_far.append(near_far)

        self.bkgd_near_far = np.array(self.bkgd_near_far)

        self.exts = np.array(scene_info['exts'])
        self.cam_points = np.linalg.inv(self.exts)[:, :3, 3].astype(np.float32)
        self.cam_dirs = np.linalg.inv(self.exts)[:, :3, :3].astype(np.float32)

        self.ixt = np.mean(self.ixts, axis=0).astype(np.float32)

        # self.input_h_w = (np.array([1920, 1080]) * self.input_ratio).astype(np.uint32).tolist()#
        H, W = self.input_h_w
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
        self.X = torch.tensor(X).cuda()
        self.Y = torch.tensor(Y).cuda()
        self.XYZ = (XYZ @ np.linalg.inv(self.ixt).T).astype(np.float32)

        self.known_cams = np.arange(16)

        self.XYZ = torch.tensor(self.XYZ).cuda()
        self.ixt = torch.tensor(self.ixt).cuda()
        self.ixts = torch.tensor(self.ixts).cuda()
        self.exts = torch.tensor(self.exts).cuda()

        self.cache = {}
        print(colored('Preparing data...', 'yellow'))
        for i in tqdm.tqdm(self.render_frames):
            self.cache_data(i)  # preloading the data into self.cache

        self.split = config['split']
        self.scene = config['scene']
        self.kwargs = config

    def read_data(self, view_id, frame_id):
        img_path = join(self.scene_root, 'images', '{:02d}'.format(view_id), '{:06d}.jpg'.format(frame_id))
        img = np.array(imageio.imread(img_path) / 255.).astype(np.float32)

        ext = np.array(self.scene_info['exts'][view_id])
        ixt = np.array(self.scene_info['ixts'][view_id]).copy()
        D = np.array(self.scene_info['Ds'][view_id]).copy()

        img = cv2.undistort(img, ixt, D)
        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            ixt[:2] *= self.input_ratio

        if self.input_h_w is not None:
            H, W, _ = img.shape
            h, w = self.input_h_w
            crop_h = int((H - h) * 0.65)  # crop more
            crop_h_ = (H - h) - crop_h
            crop_w = int((W - w) * 0.5)
            crop_w_ = W - w - crop_w
            img = img[crop_h:-crop_h_, crop_w:-crop_w_]
            ixt[1, 2] -= crop_h
            ixt[0, 2] -= crop_w
        return img, ext, ixt

    def read_data_bg(self, view_id):
        img_path = join(self.scene_root, 'bkgd', '{:02d}.jpg'.format(view_id))
        img = np.array(imageio.imread(img_path) / 255.).astype(np.float32)
        ixt = np.array(self.scene_info['ixts'][view_id]).copy()
        D = np.array(self.scene_info['Ds'][view_id]).copy()
        img = cv2.undistort(img, ixt, D)
        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            ixt[:2] *= self.input_ratio
        if self.input_h_w is not None:
            H, W, _ = img.shape
            h, w = self.input_h_w
            crop_h = int((H - h) * 0.65)  # crop more
            crop_h_ = (H - h) - crop_h
            crop_w = int((W - w) * 0.5)
            crop_w_ = W - w - crop_w
            img = img[crop_h:-crop_h_, crop_w:-crop_w_]
            ixt[1, 2] -= crop_h
            ixt[0, 2] -= crop_w
        return img

    def cache_data(self, frame):
        if frame in self.cache:
            return self.cache[frame]

        data_dict = {}
        data_dict['inps'] = []
        data_dict['bkgd'] = []
        for cam in self.known_cams:
            data_dict['inps'].append((self.read_data(cam, frame)[0] * 2. - 1.))
            data_dict['bkgd'].append((self.read_data_bg(cam)*2.-1.))
        data_dict['inps'] = np.stack(data_dict['inps']).astype(np.float32)
        data_dict['bkgd'] = np.stack(data_dict['bkgd']).astype(np.float32)
        data_dict['vertices'] = self.scene_info['bbox'][frame].astype(np.float32)
        vertices = data_dict['vertices']
        data_dict['bounds'] = np.concatenate([vertices.min(axis=0)[None], vertices.max(axis=0)[None]]).astype(np.float32)

        ret = default_convert(data_dict)
        ret = pin_memory(ret)
        self.cache[frame] = ret

    def build_rays(self, c2w, tar_ixt, scale):
        if scale != 1.:
            tar_ixt = tar_ixt.copy()
            tar_ixt[:2] *= scale
        H, W = self.input_h_w
        XYZ = self.XYZ @ c2w[:3, :3].T

        rays_o = c2w[:3, 3][None, None]
        rays_o = rays_o.repeat_interleave(H, dim=0)
        rays_o = rays_o.repeat_interleave(W, dim=1)
        rays = torch.cat((rays_o, XYZ), dim=-1)
        rays = torch.cat((rays, self.X[..., None], self.Y[..., None]), dim=-1)
        return rays.float().reshape(-1, 8), H, W

    def convert_data(self, data_dict, c2w, w2c):
        timer.logtime("")

        ext = torch.tensor(w2c).cuda()
        c2w = torch.tensor(c2w).cuda()

        input_views_num = cfg.enerf.test_input_views
        distances = np.linalg.norm(self.cam_points - c2w[:3, 3][None].cpu().numpy(), axis=-1)
        argsorts = np.argsort(distances)
        near_views = argsorts[:input_views_num]

        timer.logtime("SELECT VIEWS: {:.3f}")

        src_inps = data_dict['inps'][near_views].permute(0, 3, 1, 2).cuda()
        src_exts = self.exts[near_views]
        src_ixts = self.ixts[near_views]
        bg_src_inps = data_dict['bkgd'][near_views].permute(0, 3, 1, 2).cuda()
        ret = {'src_inps': src_inps,
               'src_exts': src_exts,
               'src_ixts': src_ixts,
               'bg_src_inps': bg_src_inps}

        tar_ext = ext.float()
        tar_ixt = self.ixt

        corners_3d = data_dict['vertices'].cuda() @ ext[:3, :3].T + ext[:3, 3:].T
        near_far = torch.tensor([corners_3d[:, 2].min(), corners_3d[:, 2].max()]).cuda()
        near_far = torch.stack([near_far, torch.tensor(self.bkgd_near_far[-1]).cuda()]).to(torch.float32)
        # near_far = torch.tensor(np.array([near_far, self.bkgd_near_far[-1]]).astype(np.float32)).cuda()

        bound_mask = data_utils.get_bound_2d_mask(corners_3d.cpu().numpy(), tar_ixt.cpu().numpy(), self.input_h_w[0], self.input_h_w[1])
        x, y, w, h = cv2.boundingRect(bound_mask.astype(np.uint8))
        w_ori, h_ori = w, h
        w = (w//32 + 1) * 32 if w % 32 != 0 or w == 0 else w
        h = (h//32 + 1) * 32 if h % 32 != 0 or h == 0 else h
        x -= (w - w_ori) // 2
        y -= (h - h_ori) // 2
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        x = self.input_h_w[1] - w if (x + w) > self.input_h_w[1] else x
        y = self.input_h_w[0] - h if (y + h) > self.input_h_w[0] else y
        xywh = torch.tensor(np.array([[x, y, w, h]]).astype(np.int32)).cuda()
        # ret.update({'meta': {'scene': '{}_{:04d}'.format(self.scene, frame_id), 'tar_view': -1, 'frame_id': frame_id}})
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        ret.update({'near_far': near_far})
        ret.update({'bbox': xywh})

        timer.logtime("MEM->GPU: {:.3f}")

        i = 1  # level 1
        scale = cfg.enerf.cas_config.render_scale[i]
        rays, H, W = self.build_rays(c2w, self.ixt, scale)
        ret.update({f'rays_{i}': rays})

        timer.logtime("BUILD RAYS: {:.3f}")

        ret = add_batch(ret)  # add a batch dimension

        timer.logtime("ADD BATCH: {:.3f}")

        return ret

    def __getitem__(self, query):
        index, c2w, w2c = query
        data_dict = self.cache_data(index)
        ret = self.convert_data(data_dict, c2w, w2c)

        return ret  # return just loaded data

    def get_camera_up_front_center(self, index=0):
        """Return the worldup, front vectors and center of the camera
        Typically used to load camera parameters
        Extrinsic Matrix: leftmost column to rightmost column: world_down cross front, world_down, front
        [
            worldup, front, center (all column vectors)
        ]

        Args:
            index(int): which camera to load
        Returns:
            worldup(np.ndarray), front(np.ndarray), center(np.ndarray)
        """
        # TODO: loading from torch might be slow?
        ext = self.exts[index].detach().cpu().numpy()
        worldup, front, center = -ext.T[:3, 1], ext.T[:3, 2], -ext[:3, :3].T @ ext[:3, 3]
        return worldup, front, center

    def get_closest_camera(self, center):
        return np.argmin(np.linalg.norm(self.cam_points - center, axis=-1))

    def get_camera_tck(self, smoothing_term=0.0):
        """Return B-spline interpolation parameters for the camera
        TODO: Actually this should be implemented as a general interpolation function
        Reference get_camera_up_front_center for the definition of worldup, front, center
        Args:
            smoothing_term(float): degree of smoothing to apply on the camera path interpolation
        """
        # - R^t @ T = cam2world translation
        # TODO: loading from torch might be slow?
        exts = self.exts.detach().cpu().numpy()  # 21, 4, 4
        # TODO: load from cam_points to avoid repeated computation
        all_cens = -np.einsum("bij,bj->bi", exts[:, :3, :3].transpose(0, 2, 1), exts[:, :3, 3]).T
        all_fros = exts[:, 2, :3].T  # (3, 21)
        all_wups = -exts[:, 1, :3].T  # (3, 21)
        cen_tck, cen_u = interpolate.splprep(all_cens, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        fro_tck, fro_u = interpolate.splprep(all_fros, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        # wup_tck, wup_u = interpolate.splprep(all_wups, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        wup_tck=None
        wup_u=None
        return cen_tck, cen_u, fro_tck, fro_u, wup_tck, wup_u

    @property
    def n_cams(self):
        return len(self.known_cams)

    def __len__(self):
        return len(self.render_frames)
