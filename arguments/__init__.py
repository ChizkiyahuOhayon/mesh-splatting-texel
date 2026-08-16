#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        self.texel_footprint_filter = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.lambda_dssim = 0.2
        # Per-face texel appearance carrier. 0 = disabled (exact original behaviour).
        # Introduced right after the restricted Delaunay retriangulation.
        self.texel_order = 0
        self.texel_lr = 0.0025
        # ResidualGate (RESEARCH_PLAN_v5): gate the normal-consistency regularizer per
        # face by g_m, the appearance-saturated cross-view-consistent photometric
        # residual (a geometric under-resolution signal). Pure PyTorch, no CUDA change.
        # resgate=False => exact original behaviour.
        self.resgate = False
        self.resgate_from_iter = 12000     # after the restricted-Delaunay transition
        self.resgate_floor = 0.1           # min phi: never fully remove smoothing (P5 mitigation)
        self.resgate_refresh = 500         # rebuild g_m every N iters (cross-view min window)
        self.resgate_norm_q = 0.95         # normalize the signal by this per-scene quantile
        self.resgate_signal = "gm"         # gm | raw | curvature  (raw/curvature = falsification negative controls)
        self.resgate_alpha = 0.5           # a pixel counts toward a face only if rendered alpha > this
        self.resgate_min_views = 3         # a face needs >= this many views in the window to get a non-neutral gate
        self.resgate_ema = 0.5             # EMA blend of g_m across refreshes (0 = hard replace) — oscillation guard
        # CGR diagnostic (E9). cgr_diag=False => exact original behaviour. Observes the
        # per-vertex photometric-gradient trajectory over a fixed-topology window and
        # dumps per-face O_i / nu_i / curvature for the offline ROC-AUC separation test.
        self.cgr_diag = False
        self.cgr_dump_iters = "4000,8000,12000,16000"  # dump the signal at these iters (several convergence stages)
        self.cgr_window = 300             # steps to accumulate the trajectory EMAs before each dump
        self.cgr_rho = 0.9                # EMA decay for mu/nu (~10-step memory)
        # Texel regularizers (pure PyTorch, no CUDA change). The texel is an additive
        # residual over the vertex colour, so:
        #   texel_l2  pulls it toward 0 -> only deviate from vertex colour when the data
        #             demands it (directly targets the ~0.87 train-test gap in E4)
        #   texel_tv  penalises within-face variance -> suppresses the high-frequency
        #             texel variation that overfits
        self.texel_l2 = 0.0
        # NOTE: this is a within-face VARIANCE penalty (deviation of a face's texels
        # from that face's mean), NOT true total variation: it uses no cross-cell
        # adjacency and does not penalise cross-face seams. Named `texel_tv` for
        # continuity with the E5 sweep; described as "within-face variance" in the paper.
        self.texel_tv = 0.0

        self.densification_interval = 500

        self.densify_from_iter = 500
        self.densify_until_iter = 10000

        self.random_background = False
        
        self.feature_lr = 0.0016 # 0.0025
        self.max_points = 4000000

        # Opacity & weight
        self.set_weight = 0.28
        self.weight_lr =  0.03
        self.lambda_weight = 1.9e-06

        # Normal loss
        self.iteration_mesh = 5000
        self.lambda_normals = 0.00005
        self.lambda_normals_super = 0.01

        self.add_percentage = 1.23

        self.set_sigma = 1.0

        # Add new triangles or vertices
        self.intervall_add_triangles = 500

        # Prune triangles and vertices
        self.prune_triangles_threshold = 0.235

        # PARAMETER SECOND STAGE
        self.lr_triangles_points_init = 0.0015

        self.start_opacity_floor = 5000

        self.start_pruning = 4000
        self.sigma_until = 30000
        self.final_opacity_iter = 24000

        self.sigma_start = 0

        # How the window hardens between `sigma_start` and `sigma_until`. All
        # schedules share the same endpoints; see sota/sigma_schedule.py.
        #   linear    -- published: linear in sigma, so most of the window's
        #                area coverage moves in the last few thousand updates
        #   coverage  -- linear in that coverage
        #   lrmatched -- linear in the vertex learning-rate budget still unspent
        self.sigma_schedule = "linear"
        # Terminal fraction of the initial vertex learning rate. It shapes both
        # the learning rate itself and the `lrmatched` hardening path, so the two
        # cannot drift apart.
        self.lr_triangles_points_decay = 0.01

        self.splitt_large_triangles = 100
        self.start_upsampling = 20000
        self.upscaling_factor = 2
        # Supersampling factor for the last training phase, and the rate the
        # model is deployed at. 4 is the published behaviour: sixteen samples
        # per rendered pixel. SAC-G0 asks what that last factor is worth when
        # the model is trained at the rate it will be rendered at.
        self.final_scaling = 4
        # Supersampling factor for the post-training cleanup, which deletes
        # faces whose maximum blending weight never exceeds 0.5. That maximum
        # is taken per pixel, so the factor decides how many chances each face
        # gets to survive. 0 keeps whatever factor training ended with, which
        # is the published behaviour; setting it makes the criterion
        # independent of the training schedule.
        self.cleanup_scaling = 0

        self.size_probs_zero = 7.5e-05
        self.size_probs_zero_image_space = 0.0

        self.prune_size = 1400

        self.lambda_vertex = 0.00025
        self.max_diff_threshold = 0.5
        self.start_vertex_opt = 12000

        self.lamba_depth = 0.05

        self.depth_lambda_init = 0.01
        self.depth_lambda_final = 0.001

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def update_indoor(params):
    params.add_percentage = 1.27
    params.densify_from_iter = 1000
    params.densify_until_iter = 10000
    params.feature_lr = 0.004
    params.size_probs_zero = 0.0
    params.splitt_large_triangles = 500
    params.start_pruning = 3000
    params.weight_lr = 0.05
    params.lambda_weight = 0.0
    params.lambda_normals = 0.00001
    params.lambda_normals_super = 0.01
    params.prune_size = 1300
    params.lambda_vertex = 0.00025
    params.depth_lambda_init = 0.0
    params.depth_lambda_final = 0.0
    params.iteration_mesh = 12000
    return params
