# run.py (修改版，严格参照您的版本)

import torch
import argparse
import pickle
import warnings
import os
import random
import time
import numpy as np

warnings.filterwarnings("ignore")

# 保持您原有的、可以正常工作的导入方式
from test_functions import * 
# MixedFixedAlphaStableKernel 的导入是为了在构造核实例时进行类型检查（可选，但好习惯）
# 但为了严格遵循最小化改动，我们可以暂时不导入
# from botorch.models.kernels.stable_kernel import MixedFixedAlphaStableKernel 
from utils.utils import get_next_points

parser = argparse.ArgumentParser('Run BayesOpt Experiments')
parser.add_argument('--function_name', type=str, default='branin2', help='objective function')
parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
parser.add_argument('--n_init', type=int, default=5, help='number of initial random points')
# --- 在 help 文本中添加新的核选项 ---
parser.add_argument('--kernel', type=str, default='mixstable_fixed_alpha', help='choice of kernel (e.g., rbf, csm, gsm, mix, mixstable_fixed_alpha, learnable_alpha_stable)')
parser.add_argument('--acq', type=str, default='ei', help='choice of the acquisition function')
parser.add_argument('--n_mixture', type=int, default=7, help='number of mixtures for gsm, csm, and learnable_alpha_stable kernels')
parser.add_argument('--n_mixture1', type=int, default=2, help='number of Cauchy mixtures for mix kernel')
parser.add_argument('--n_mixture2', type=int, default=2, help='number of Gaussian mixtures for mix kernel')
parser.add_argument('--alphas', type=float, nargs='+', default=[1.0, 2.0], help='List of fixed alpha values for MixedFixedAlphaStableKernel')
parser.add_argument('--seed', type=int, default=1, help='random seed')


args = parser.parse_args()
options = vars(args)
print(options)

seed_list = range(0, args.seed, 1)
best_all = []
time_start = time.time()
for seed in seed_list:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("----------------------------Running seed {}----------------------------".format(seed))
    # --- 保持您原有的函数实例化逻辑 ---
    if args.function_name == 'branin2':
        func = Branin()
    elif args.function_name == 'hartmann3':
        func = Hartmann3()
    elif args.function_name == 'griewank5':
        func = Griewank(dim=5)
    elif args.function_name == 'hartmann6':
        func = Hartmann6()
    elif args.function_name == 'exp5':
        func = Exponential(dim=5)
    elif args.function_name == 'exp10':
        func = Exponential(dim=10)
    elif args.function_name == 'rosen20':
        func = Rosenbrock(dim=20)
    elif args.function_name == 'levy30':
        func = Levy(dim=30)
    elif args.function_name == 'robot3':
        gpos = 10 * torch.randn(1, 2) - 5
        func = Robot3(gpos[0][0], gpos[0][1])
    elif args.function_name == 'robot4':
        gpos = 10 * torch.randn(1, 2) - 5
        func = Robot4(gpos[0][0], gpos[0][1])
    elif args.function_name == 'portfolio5':
        func = PortfolioSurrogate()
    elif args.function_name == 'XGBoost9':
        func = XGBoost_HPO()
    elif args.function_name == 'XGBoost14':
        func = XGBoost_HPO_14D()
    elif args.function_name == 'LightGBM16':
        func = LightGBM_HPO()
    elif args.function_name == 'SVM3':
        func = SVM_HPO()
    else:
        raise ValueError('Unrecognised problem %s' % args.function_name)

    # --- 保持您原有的边界和数据生成逻辑 ---
    d = func.dim
    lb = func.lb
    ub = func.ub
    bounds = torch.stack((lb, ub))
    optimum = func.min

    init_x = torch.rand(args.n_init, d, dtype=torch.float32)
    init_x = bounds[0] + (bounds[1] - bounds[0]) * init_x
    if args.function_name == 'robot3' or args.function_name == 'robot4':
        init_x = torch.mean(init_x, dim=0, keepdim=True)
    init_y = func.eval(init_x)
    best_init_y = init_y.min().item()

    n_iterations = args.n_iter
    best_result = [best_init_y]
    try:
        # --- 核函数处理：与之前对齐，不在此处实例化 ---
        kernel_to_pass = args.kernel
        print(f"Passing kernel identifier to GP: {kernel_to_pass}")
        if kernel_to_pass == 'mixstable_fixed_alpha':
            if args.alphas is None or len(args.alphas) == 0:
                raise ValueError("`--alphas` argument is required for kernel 'mixstable_fixed_alpha'")
            print(f"Will use alphas: {args.alphas}")
        # --- 新增：为 learnable_alpha_stable 打印信息 ---
        elif kernel_to_pass == 'learnable_alpha_stable':
            print(f"Will use learnable alpha kernel with Q={args.n_mixture} components.")

        # --- BO 循环：保持您原有的调用结构 ---
        for i in range(n_iterations):
            print(f"Number of iterations done: {i}")
            
            alphas_to_pass = args.alphas if args.kernel == 'mixstable_fixed_alpha' else None
            new_candidates = get_next_points(
                acq=args.acq,
                kernel=kernel_to_pass,       # 传递核名称字符串
                n_mixture=args.n_mixture,    # 传递 Q
                init_x=init_x,
                init_y=init_y,
                best_init_y=best_init_y,
                bounds=bounds,
                n_points=1,
                n_mixture1=args.n_mixture1,
                n_mixture2=args.n_mixture2,
                alphas=alphas_to_pass         # 传递固定的 alphas 列表或 None
            )
            new_results = func.eval(new_candidates)

            print(f"New candidates are: {new_candidates}, {new_results}")
            init_x = torch.cat([init_x, new_candidates])
            init_y = torch.cat([init_y, new_results])

            best_init_y = init_y.min().item()
            print(f"f_min: {best_init_y}")
            best_result.append(best_init_y)
        best_all.append(best_result)
    except Exception as e:
        print(f"Seed {seed} failed: {e}")
        import traceback
        traceback.print_exc()


time_end = time.time()
running_time = (time_end - time_start)/max(1, len(seed_list))
print(f"Running time for {args.function_name}: {running_time} seconds")

current_dir = os.path.dirname(os.path.abspath(__file__))
# --- 文件名生成逻辑：添加对新核的支持 ---
if args.kernel == 'csm' or args.kernel == 'gsm':
    filename = f"{args.function_name}_{args.kernel}{args.n_mixture}_{args.acq}.pkl"
elif args.kernel == 'mix':
    filename = f"{args.function_name}_c{args.n_mixture1}g{args.n_mixture2}_{args.acq}.pkl"
elif args.kernel == 'mixstable_fixed_alpha':
    alpha_str = '_'.join([f"{a:.1f}".replace('.', 'p') for a in args.alphas])
    filename = f"{args.function_name}_mixstable_a{alpha_str}_{args.acq}.pkl"
# --- 新增 elif 分支 ---
elif args.kernel == 'learnable_alpha_stable':
    filename = f"{args.function_name}_learnable_alpha_q{args.n_mixture}_{args.acq}.pkl"
# --- 结束新增 ---
else:
    filename = f"{args.function_name}_{args.kernel}_{args.acq}.pkl"

save_dir = os.path.join(current_dir, 'exp_res_best', 'pkl')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, filename)
print(f"Saving results to: {file_path}")
with open(file_path, 'wb') as f:
    pickle.dump(best_all, f)