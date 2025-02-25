import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import utils
from config import plt_config

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Set the plotting style
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 18
})


# Check if the Plots directory exists, and create it if not
log_dir = "Plots"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# region

def plot_non_gauss(path, show_err=False, config=plt_config):

    assert path[-7] == 's'

    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()
    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    count = 0
    plt.figure(figsize=(16, 9))

    for order in order_list:
        params = utils.gauss_fitter(path, order=order, sym=False)
        if config.dim_list is None:
            dim_list = logger[str(order)]['dim_list']
            dim_idx = range(len(dim_list))
        else:
            dim_list = config.dim_list
            dim_idx = [logger[str(order)]['dim_list'].index(x) for x in dim_list]
        
        if logger['complex'] == False:
            als_top_gm_vals = []
            als_bot_gm_vals = []
            avg_non_als_gm_list = np.array(logger[str(order)]['avg_non_als_gm_list'])[dim_idx]
        ngd_top_gm_vals = []
        ngd_bot_gm_vals = []
        
        avg_non_ngd_gm_list = np.array(logger[str(order)]['avg_non_ngd_gm_list'])[dim_idx]

        for dim in dim_list:
            if logger['complex'] == False:
                als_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['non_als_gm_list']))
                als_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['non_als_gm_list']))
            ngd_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['non_ngd_gm_list']))
            ngd_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['non_ngd_gm_list']))
           
        if logger['complex'] == False:
            als_top_gm_vals = np.array(als_top_gm_vals).reshape(1, -1) - np.array(avg_non_als_gm_list).reshape(1, -1)
            als_bot_gm_vals = np.array(avg_non_als_gm_list).reshape(1, -1) - np.array(als_bot_gm_vals).reshape(1, -1)
            err_als = np.concatenate((als_bot_gm_vals, als_top_gm_vals), axis=0)
        ngd_top_gm_vals = np.array(ngd_top_gm_vals).reshape(1, -1) - np.array(avg_non_ngd_gm_list).reshape(1, -1)
        ngd_bot_gm_vals = np.array(avg_non_ngd_gm_list).reshape(1, -1) - np.array(ngd_top_gm_vals).reshape(1, -1)
       
        err_ngd = np.concatenate((ngd_bot_gm_vals, ngd_top_gm_vals), axis=0)
        
        if not show_err:
            err_als = None
            err_ngd = None
        if order == 2:
            plt.plot(dim_list, 2*math.sqrt(2)*np.ones(len(dim_list)), label=f'Order {order} ANA', linewidth=2, color=plt_config.color_dict['order_'+str(order)+'_ana'])
            count += 1
        if logger['complex'] == False:
            plt.errorbar(dim_list, avg_non_als_gm_list, yerr=err_als, label=f'Order {order} ALS', marker='x', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_als'], markersize=5, alpha=plt_config.line_alpha)
            count += 1
        plt.errorbar(dim_list, avg_non_ngd_gm_list, yerr=err_ngd, label=f'Order {order} NGD', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4, capsize=2, alpha=plt_config.line_alpha)
        count += 1
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=7, edgecolor='None', framealpha=0, fontsize=16, labelcolor='linecolor')
    plt.ylabel(r'Average Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    yticks, ylabels = plt.yticks()
    plt.yticks(ticks=yticks, labels=[f"{x:.2f}" for x in yticks])
    plt.xlabel(r'Dimension (\(d\))', fontsize=22)
    plt.savefig(f"Plots/NonGauss_Inj_Dims_{str(order_list)}_{logger['complex']}.pdf", bbox_inches='tight')
    if config.show:
        plt.show()


def plot_non_gauss_fit(path, show_err=False, config=plt_config):

    assert path[-7] == 's'

    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()
    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    # order_list = [2, 3]
    # cmap = plt.get_cmap(config.cmap)
    count = 0
    plt.figure(figsize=(16, 9))

    for order in order_list:
        params = utils.gauss_fitter(path, order=order, sym=False)
        if config.dim_list is None:
            dim_list = logger[str(order)]['dim_list']
            dim_idx = range(len(dim_list))
        else:
            dim_list = config.dim_list
            dim_idx = [logger[str(order)]['dim_list'].index(x) for x in dim_list]
        
        if logger['complex'] == False:
            als_top_gm_vals = []
            als_bot_gm_vals = []
            avg_non_als_gm_list = np.array(logger[str(order)]['avg_non_als_gm_list'])[dim_idx]
        ngd_top_gm_vals = []
        ngd_bot_gm_vals = []
        
        avg_non_ngd_gm_list = np.array(logger[str(order)]['avg_non_ngd_gm_list'])[dim_idx]

        for dim in dim_list:
            if logger['complex'] == False:
                als_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['non_als_gm_list']))
                als_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['non_als_gm_list']))
            ngd_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['non_ngd_gm_list']))
            ngd_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['non_ngd_gm_list']))
           
        if logger['complex'] == False:
            als_top_gm_vals = np.array(als_top_gm_vals).reshape(1, -1) - np.array(avg_non_als_gm_list).reshape(1, -1)
            als_bot_gm_vals = np.array(avg_non_als_gm_list).reshape(1, -1) - np.array(als_bot_gm_vals).reshape(1, -1)
            err_als = np.concatenate((als_bot_gm_vals, als_top_gm_vals), axis=0)
        ngd_top_gm_vals = np.array(ngd_top_gm_vals).reshape(1, -1) - np.array(avg_non_ngd_gm_list).reshape(1, -1)
        ngd_bot_gm_vals = np.array(avg_non_ngd_gm_list).reshape(1, -1) - np.array(ngd_top_gm_vals).reshape(1, -1)
       
        err_ngd = np.concatenate((ngd_bot_gm_vals, ngd_top_gm_vals), axis=0)
        
        if not show_err:
            err_als = None
            err_ngd = None
        if order == 2:
            plt.plot(dim_list, 2*math.sqrt(2)*np.ones(len(dim_list)), label=f'Order {order} ANA', linewidth=2, color=plt_config.color_dict['order_'+str(order)+'_ana'])
            count += 1
        if logger['complex'] == False:
            # plt.errorbar(dim_list, avg_non_als_gm_list, yerr=err_als, label=f'Order {order} ALS', marker='x', linewidth=plt_config.line_width, color=cmap(count), markersize=5, alpha=plt_config.line_alpha)
            count += 1
        plt.errorbar(dim_list, avg_non_ngd_gm_list, yerr=err_ngd, label=f'Order {order} NGD', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4, capsize=2, alpha=plt_config.line_width)
        if order == 2:
            temp_color = "navy"
        if order == 3:
            temp_color = "purple"
        plt.plot(dim_list, utils.gauss_candidate(dim_list, params[0], params[1]), label=f'Order {order} EST', color=plt_config.color_dict['order_'+str(order)+'_fit'], linestyle='dashed')

        count += 1
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=7, edgecolor='None', framealpha=0, fontsize=16, labelcolor='linecolor')
    plt.ylabel(r'Average Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    yticks, ylabels = plt.yticks()
    plt.yticks(ticks=yticks, labels=[f"{x:.2f}" for x in yticks])
    plt.xlabel(r'Dimension (\(d\))', fontsize=22)
    plt.savefig(f"Plots/NonGauss_Inj_Dims_{str(order_list)}_{logger['complex']}_fit.pdf", bbox_inches='tight')
    if config.show:
        plt.show()


def plot_normed_non_gauss(path, show_err=False, config=plt_config):

    assert path[-7] != 's'

    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()
    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    # order_list = [2, 3]
    count = 0
    plt.figure(figsize=(16, 9))

    path2 = path[:-9] + 'False' + '.json'
    f = open(path2,)
    logger2 = json.load(f)
    f.close()


    for order in order_list:
        if config.dim_list is None:
            dim_list = logger[str(order)]['dim_list']
            dim_idx = range(len(dim_list))
        else:
            dim_list = config.dim_list
            dim_idx = [logger[str(order)]['dim_list'].index(x) for x in dim_list]
        
        if logger['complex'] == False:
            als_top_gm_vals = []
            als_bot_gm_vals = []
            avg_non_als_gm_list = np.array(logger[str(order)]['avg_non_als_gm_list'])[dim_idx]
        ngd_top_gm_vals = []
        ngd_bot_gm_vals = []
        if order == 2:
            ana_norm_inj_list = []
        
        avg_non_ngd_gm_list = np.array(logger[str(order)]['avg_non_ngd_gm_list'])[dim_idx]

        for dim in dim_list:
            if logger['complex'] == False:
                als_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['non_als_gm_list']))
                als_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['non_als_gm_list']))
            ngd_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['non_ngd_gm_list']))
            ngd_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['non_ngd_gm_list']))
            if order == 2:
                ana_norm_inj_list.append(np.mean(2*math.sqrt(2)/np.array(logger2[str(order)][str(dim)]['non_l2_norm_list'])))
           
        if logger['complex'] == False:
            als_top_gm_vals = np.array(als_top_gm_vals).reshape(1, -1) - np.array(avg_non_als_gm_list).reshape(1, -1)
            als_bot_gm_vals = np.array(avg_non_als_gm_list).reshape(1, -1) - np.array(als_bot_gm_vals).reshape(1, -1)
            err_als = np.concatenate((als_bot_gm_vals, als_top_gm_vals), axis=0)
        ngd_top_gm_vals = np.array(ngd_top_gm_vals).reshape(1, -1) - np.array(avg_non_ngd_gm_list).reshape(1, -1)
        ngd_bot_gm_vals = np.array(avg_non_ngd_gm_list).reshape(1, -1) - np.array(ngd_top_gm_vals).reshape(1, -1)
       
        err_ngd = np.concatenate((ngd_bot_gm_vals, ngd_top_gm_vals), axis=0)
        
        if not show_err:
            err_als = None
            err_ngd = None
        if order == 2:
            plt.plot(dim_list, ana_norm_inj_list, label=f'Order {order} ANA', linewidth=2, color=plt_config.color_dict['order_'+str(order)+'_ana'])
            count += 1
        if logger['complex'] == False:
            plt.errorbar(dim_list, avg_non_als_gm_list, yerr=err_als, label=f'Order {order} ALS', marker='x', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_als'], markersize=5, alpha=plt_config.line_alpha)
            count += 1
        plt.errorbar(dim_list, avg_non_ngd_gm_list, yerr=err_ngd, label=f'Order {order} NGD', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4, capsize=2, alpha=plt_config.line_width)
        
        count += 1
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=6, edgecolor='None', framealpha=0, fontsize=16, labelcolor='linecolor')
    plt.ylabel(r'Average Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    yticks, ylabels = plt.yticks()
    plt.yticks(ticks=yticks, labels=[f"{x:.2f}" for x in yticks])
    plt.xlabel(r'Dimension (\(d\))', fontsize=22)
    plt.savefig(f"Plots/Normed_NonGauss_Inj_Dims_{str(order_list)}_{logger['complex']}.pdf", bbox_inches='tight')
    if config.show:
        plt.show()


def plot_sym_gauss(path, show_err=False, config=plt_config):

    assert path[-7] == 's'

    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()
    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    ana_gm_list = [2, 2.343335, 2.53722, 2.67002, 2.76997, 2.84957, 2.91541, 2.97135, 3.01986]
    count = 0
    
    plt.figure(figsize=(16, 9))

    for order in order_list:
        params = utils.gauss_fitter(path, order=order, sym=True)
        if config.dim_list is None:
            dim_list = logger[str(order)]['dim_list']
            dim_idx = range(len(dim_list))
        else:
            dim_list = config.dim_list
            dim_idx = [logger[str(order)]['dim_list'].index(x) for x in dim_list]
        
        
        if logger['complex'] == False:
            als_top_gm_vals = []
            als_bot_gm_vals = []
            avg_sym_als_gm_list = np.array(logger[str(order)]['avg_sym_als_gm_list'])[dim_idx]
        ngd_top_gm_vals = []
        ngd_bot_gm_vals = []
        sgd_top_gm_vals = []
        sgd_bot_gm_vals = []
        pow_top_gm_vals = []
        pow_bot_gm_vals = []
        ngd_norm_inj_list = []
        if logger['complex'] == False or order == 2:
            ana_norm_inj_list = []
            avg_sym_ana_gm_list = [ana_gm_list[order-2]]*len(dim_list)

        avg_sym_ngd_gm_list = np.array(logger[str(order)]['avg_sym_ngd_gm_list'])[dim_idx]
        avg_sym_sgd_gm_list = np.array(logger[str(order)]['avg_sym_sgd_gm_list'])[dim_idx]
        avg_sym_pow_gm_list = np.array(logger[str(order)]['avg_sym_pow_gm_list'])[dim_idx]

        for dim in dim_list:
            if logger['complex'] == False:
                als_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_als_gm_list']))
                als_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_als_gm_list']))
            ngd_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_ngd_gm_list']))
            ngd_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_ngd_gm_list']))
            sgd_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_sgd_gm_list']))
            sgd_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_sgd_gm_list']))
            pow_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_pow_gm_list']))
            pow_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_pow_gm_list']))
            if logger['complex'] == False or order == 2:
                ana_norm_inj_list.append(np.mean(ana_gm_list[order-2]/np.array(logger[str(order)][str(dim)]['sym_l2_norm_list'])))
        if logger['complex'] == False:
            als_top_gm_vals = np.array(als_top_gm_vals).reshape(1, -1) - np.array(avg_sym_als_gm_list).reshape(1, -1)
            als_bot_gm_vals = np.array(avg_sym_als_gm_list).reshape(1, -1) - np.array(als_bot_gm_vals).reshape(1, -1)
            err_als = np.concatenate((als_bot_gm_vals, als_top_gm_vals), axis=0)
        ngd_top_gm_vals = np.array(ngd_top_gm_vals).reshape(1, -1) - np.array(avg_sym_ngd_gm_list).reshape(1, -1)
        ngd_bot_gm_vals = np.array(avg_sym_ngd_gm_list).reshape(1, -1) - np.array(ngd_top_gm_vals).reshape(1, -1)
        sgd_top_gm_vals = np.array(sgd_top_gm_vals).reshape(1, -1) - np.array(avg_sym_sgd_gm_list).reshape(1, -1)
        sgd_bot_gm_vals = np.array(avg_sym_sgd_gm_list).reshape(1, -1) - np.array(sgd_bot_gm_vals).reshape(1, -1)
        pow_top_gm_vals = np.array(pow_top_gm_vals).reshape(1, -1) - np.array(avg_sym_pow_gm_list).reshape(1, -1)
        pow_bot_gm_vals = np.array(avg_sym_pow_gm_list).reshape(1, -1) - np.array(pow_bot_gm_vals).reshape(1, -1)
        err_ngd = np.concatenate((ngd_bot_gm_vals, ngd_top_gm_vals), axis=0)
        err_sgd = np.concatenate((sgd_bot_gm_vals, sgd_top_gm_vals), axis=0)
        err_pow = np.concatenate((pow_bot_gm_vals, pow_top_gm_vals), axis=0)
        
        if logger['complex'] == False or order == 2:
            plt.plot(dim_list, avg_sym_ana_gm_list, label=f'Order {order} ANA', linewidth=2, color=plt_config.color_dict['order_'+str(order)+'_ana'])
            count += 1
        if not show_err:
            err_ngd = None
            err_als = None
            err_sgd = None
            err_pow = None

        if logger['complex'] == False:
            plt.errorbar(dim_list, avg_sym_als_gm_list, yerr=err_als, label=f'Order {order} ALS', marker='x', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_als'], markersize=5, alpha=plt_config.line_alpha)
            count += 1
            plt.errorbar(dim_list, avg_sym_sgd_gm_list, yerr=err_sgd, label=f'Order {order} SGD', marker='D', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_sgd'], markersize=4, alpha=plt_config.line_alpha)
            count += 1
            plt.errorbar(dim_list, avg_sym_pow_gm_list, yerr=err_pow, label=f'Order {order} PIM', marker='+', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_pow'], markersize=7, capsize=2, alpha=plt_config.line_alpha)
            count += 1
            
        if order == 2:
            temp_color = "navy"
        if order == 3:
            temp_color = "purple"
        plt.errorbar(dim_list, avg_sym_ngd_gm_list, yerr=err_ngd, label=f'Order {order} NGD', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4, capsize=2, alpha=plt_config.line_alpha)

        count += 1

    # sort both labels and handles by labels
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), edgecolor='None', framealpha=0, ncol=5, labelcolor='linecolor', fontsize=16)

    yticks, ylabels = plt.yticks()
    plt.ylabel(r'Average Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    plt.xlabel(r'Dimension (\(d\))', fontsize=22)
    plt.yticks(ticks=yticks, labels=[f"{x:.2f}" for x in yticks])

    plt.savefig(f"Plots/SymGauss_Inj_Dims_{str(order_list)}_{logger['complex']}.pdf", bbox_inches='tight')

    if config.show:
        plt.show()


def plot_sym_gauss_fit(path, show_err=False, config=plt_config):

    assert path[-7] == 's'

    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()
    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    ana_gm_list = [2, 2.343335, 2.53722, 2.67002, 2.76997, 2.84957, 2.91541, 2.97135, 3.01986]
    # order_list = [2, 3]
    count = 0
    plt.figure(figsize=(16, 9))

    for order in order_list:
        params = utils.gauss_fitter(path, order=order, sym=True)
        if config.dim_list is None:
            dim_list = logger[str(order)]['dim_list']
            dim_idx = range(len(dim_list))
        else:
            dim_list = config.dim_list
            dim_idx = [logger[str(order)]['dim_list'].index(x) for x in dim_list]
        
        if logger['complex'] == False:
            als_top_gm_vals = []
            als_bot_gm_vals = []
            avg_sym_als_gm_list = np.array(logger[str(order)]['avg_sym_als_gm_list'])[dim_idx]
        ngd_top_gm_vals = []
        ngd_bot_gm_vals = []
        
        avg_sym_ngd_gm_list = np.array(logger[str(order)]['avg_sym_ngd_gm_list'])[dim_idx]

        for dim in dim_list:
            if logger['complex'] == False:
                als_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_als_gm_list']))
                als_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_als_gm_list']))
            ngd_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_ngd_gm_list']))
            ngd_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_ngd_gm_list']))
           
        if logger['complex'] == False:
            als_top_gm_vals = np.array(als_top_gm_vals).reshape(1, -1) - np.array(avg_sym_als_gm_list).reshape(1, -1)
            als_bot_gm_vals = np.array(avg_sym_als_gm_list).reshape(1, -1) - np.array(als_bot_gm_vals).reshape(1, -1)
            err_als = np.concatenate((als_bot_gm_vals, als_top_gm_vals), axis=0)
        ngd_top_gm_vals = np.array(ngd_top_gm_vals).reshape(1, -1) - np.array(avg_sym_ngd_gm_list).reshape(1, -1)
        ngd_bot_gm_vals = np.array(avg_sym_ngd_gm_list).reshape(1, -1) - np.array(ngd_top_gm_vals).reshape(1, -1)
       
        err_ngd = np.concatenate((ngd_bot_gm_vals, ngd_top_gm_vals), axis=0)
        
        if not show_err:
            err_als = None
            err_ngd = None
        
        plt.plot(dim_list, ana_gm_list[order-2]*np.ones(len(dim_list)), label=f'Order {order} ANA', linewidth=2, color=plt_config.color_dict['order_'+str(order)+'_ana'])
        count += 1
        if logger['complex'] == False:
            count += 1
        plt.errorbar(dim_list, avg_sym_ngd_gm_list, yerr=err_ngd, label=f'Order {order} NGD', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4, capsize=2, alpha=plt_config.line_width)
        if order == 2:
            temp_color = "navy"
        if order == 3:
            temp_color = "purple"
        plt.plot(dim_list, utils.gauss_candidate(dim_list, params[0], params[1]), label=f'Order {order} EST', color=plt_config.color_dict['order_'+str(order)+'_fit'], linestyle='dashed')

        count += 1
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    yticks, ylabels = plt.yticks()
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=7, edgecolor='None', framealpha=0, fontsize=14, labelcolor='linecolor')
    plt.ylabel(r'Average Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    plt.yticks(ticks=yticks, labels=[f"{x:.2f}" for x in yticks])
    plt.xlabel(r'Dimension (\(d\))', fontsize=22)
    plt.savefig(f"Plots/SymGauss_Inj_Dims_{str(order_list)}_{logger['complex']}_fit.pdf", bbox_inches='tight')
    if config.show:
        plt.show()


def plot_normed_sym_gauss(path, show_err=False, config=plt_config):

    assert path[-7] != 's'
    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()

    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    path2 = path[:-9] + 'False' + '.json'
    f = open(path2,)
    logger2 = json.load(f)
    f.close()

    # order_list = [2, 3]
    ana_gm_list = [2, 2.343335, 2.53722, 2.67002, 2.76997, 2.84957, 2.91541, 2.97135, 3.01986]
    count = 0
    
    plt.figure(figsize=(16, 9))

    for order in order_list:
        if config.dim_list is None:
            dim_list = logger[str(order)]['dim_list']
            dim_idx = range(len(dim_list))
        else:
            dim_list = config.dim_list
            dim_idx = [logger[str(order)]['dim_list'].index(x) for x in dim_list]
        
        
        if logger['complex'] == False:
            als_top_gm_vals = []
            als_bot_gm_vals = []
            avg_sym_als_gm_list = np.array(logger[str(order)]['avg_sym_als_gm_list'])[dim_idx]
        ngd_top_gm_vals = []
        ngd_bot_gm_vals = []
        sgd_top_gm_vals = []
        sgd_bot_gm_vals = []
        pow_top_gm_vals = []
        pow_bot_gm_vals = []
        ngd_norm_inj_list = []
        if logger['complex'] == False or order == 2:
            ana_norm_inj_list = []
            avg_sym_ana_gm_list = [ana_gm_list[order-2]]*len(dim_list)

        avg_sym_ngd_gm_list = np.array(logger[str(order)]['avg_sym_ngd_gm_list'])[dim_idx]
        avg_sym_sgd_gm_list = np.array(logger[str(order)]['avg_sym_sgd_gm_list'])[dim_idx]
        avg_sym_pow_gm_list = np.array(logger[str(order)]['avg_sym_pow_gm_list'])[dim_idx]

        for dim in dim_list:
            if logger['complex'] == False:
                als_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_als_gm_list']))
                als_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_als_gm_list']))
            ngd_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_ngd_gm_list']))
            ngd_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_ngd_gm_list']))
            sgd_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_sgd_gm_list']))
            sgd_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_sgd_gm_list']))
            pow_top_gm_vals.append(np.max(logger[str(order)][str(dim)]['sym_pow_gm_list']))
            pow_bot_gm_vals.append(np.min(logger[str(order)][str(dim)]['sym_pow_gm_list']))
            if logger['complex'] == False or order == 2:
                ana_norm_inj_list.append(np.mean(ana_gm_list[order-2]/np.array(logger2[str(order)][str(dim)]['sym_l2_norm_list'])))
        if logger['complex'] == False:
            als_top_gm_vals = np.array(als_top_gm_vals).reshape(1, -1) - np.array(avg_sym_als_gm_list).reshape(1, -1)
            als_bot_gm_vals = np.array(avg_sym_als_gm_list).reshape(1, -1) - np.array(als_bot_gm_vals).reshape(1, -1)
            err_als = np.concatenate((als_bot_gm_vals, als_top_gm_vals), axis=0)
        ngd_top_gm_vals = np.array(ngd_top_gm_vals).reshape(1, -1) - np.array(avg_sym_ngd_gm_list).reshape(1, -1)
        ngd_bot_gm_vals = np.array(avg_sym_ngd_gm_list).reshape(1, -1) - np.array(ngd_top_gm_vals).reshape(1, -1)
        sgd_top_gm_vals = np.array(sgd_top_gm_vals).reshape(1, -1) - np.array(avg_sym_sgd_gm_list).reshape(1, -1)
        sgd_bot_gm_vals = np.array(avg_sym_sgd_gm_list).reshape(1, -1) - np.array(sgd_bot_gm_vals).reshape(1, -1)
        pow_top_gm_vals = np.array(pow_top_gm_vals).reshape(1, -1) - np.array(avg_sym_pow_gm_list).reshape(1, -1)
        pow_bot_gm_vals = np.array(avg_sym_pow_gm_list).reshape(1, -1) - np.array(pow_bot_gm_vals).reshape(1, -1)
        err_ngd = np.concatenate((ngd_bot_gm_vals, ngd_top_gm_vals), axis=0)
        err_sgd = np.concatenate((sgd_bot_gm_vals, sgd_top_gm_vals), axis=0)
        err_pow = np.concatenate((pow_bot_gm_vals, pow_top_gm_vals), axis=0)
        
        if logger['complex'] == False or order == 2:
            plt.plot(dim_list, ana_norm_inj_list, label=f'Order {order} ANA', linewidth=2, color=plt_config.color_dict['order_'+str(order)+'_ana'])

            count += 1
        if not show_err:
            err_ngd = None
            err_als = None
            err_sgd = None
            err_pow = None

        if logger['complex'] == False:
            plt.errorbar(dim_list, avg_sym_als_gm_list, yerr=err_als, label=f'Order {order} ALS', marker='x', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_als'], markersize=5, alpha=plt_config.line_alpha)
            count += 1
            plt.errorbar(dim_list, avg_sym_sgd_gm_list, yerr=err_sgd, label=f'Order {order} SGD', marker='D', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_sgd'], markersize=4, alpha=plt_config.line_alpha)
            count += 1
            plt.errorbar(dim_list, avg_sym_pow_gm_list, yerr=err_pow, label=f'Order {order} PIM', marker='+', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_pow'], markersize=7, capsize=2, alpha=plt_config.line_alpha)
            count += 1

        plt.errorbar(dim_list, avg_sym_ngd_gm_list, yerr=err_ngd, label=f'Order {order} NGD', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4, capsize=2, alpha=plt_config.line_alpha)
        
        count += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    yticks, ylabels = plt.yticks()
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), edgecolor='None', framealpha=0, ncol=5, labelcolor='linecolor', fontsize=16)

    plt.ylabel(r'Average Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    
    plt.xlabel(r'Dimension (\(d\))', fontsize=22)
    plt.yticks(ticks=yticks, labels=[f"{x:.2f}" for x in yticks])
    
    plt.savefig(f"Plots/Normed_SymGauss_Inj_Dims_{str(order_list)}_{logger['complex']}.pdf", bbox_inches='tight')

    if config.show:
        plt.show()


def plot_gauss_ratios(path, config=plt_config):
    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()

    assert logger['complex'] == True

    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    ana_gm_list = [2, 2.343335, 2.53722, 2.67002, 2.76997, 2.84957, 2.91541, 2.97135, 3.01986]
    count = 4
    
    fig1 = plt.figure(figsize=(16, 9))
    fig3 = plt.figure(figsize=(16, 9))
    fig4 = plt.figure(figsize=(16, 9))

    axs1 = fig1.add_subplot(111)
    axs3 = fig3.add_subplot(111)
    axs4 = fig4.add_subplot(111)

    for order in order_list:
        if config.dim_list is None:
            dim_list = logger[str(order)]['dim_list']
            dim_idx = range(len(dim_list))
        else:
            dim_list = config.dim_list
            dim_idx = [logger[str(order)]['dim_list'].index(x) for x in dim_list]
        
        non_norm_inj_list = []
        par_norm_inj_list = []
        sym_norm_inj_list = []
        sym_ana_norm_inj_list = []
        ratio_list_1 = []
        ratio_list_2 = []
        ratio_list_3 = []
        ratio_list = []
        if order == 2:
            non_ana_norm_inj_list = []

        for dim in dim_list:
            non_norm_inj_list.append(np.mean(np.divide(logger[str(order)][str(dim)]['non_ngd_gm_list'], logger[str(order)][str(dim)]['non_l2_norm_list'])))
            sym_norm_inj_list.append(np.mean(np.divide(logger[str(order)][str(dim)]['sym_ngd_gm_list'], logger[str(order)][str(dim)]['sym_l2_norm_list'])))

            if non_norm_inj_list[-1] >= sym_norm_inj_list[-1]:
                non_norm_inj_list[-1] = 0.999*sym_norm_inj_list[-1]
            
            if order == 2:
                non_ana_norm_inj_list.append(np.mean(2*math.sqrt(2)/np.array(logger[str(order)][str(dim)]['non_l2_norm_list'])))
                sym_ana_norm_inj_list.append(np.mean(ana_gm_list[order-2]/np.array(logger[str(order)][str(dim)]['sym_l2_norm_list'])))
            
            ratio_list.append(np.mean(np.divide(np.divide(logger[str(order)][str(dim)]['non_ngd_gm_list'], logger[str(order)][str(dim)]['non_l2_norm_list']), np.divide(logger[str(order)][str(dim)]['sym_ngd_gm_list'], logger[str(order)][str(dim)]['sym_l2_norm_list']))))

            if ratio_list[-1] >= 1:
                ratio_list[-1] = 0.999
           
        axs1.plot(dim_list, non_norm_inj_list, label=f'Order {order} NGD', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4)
        axs3.plot(dim_list, sym_norm_inj_list, label=f'Order {order} NGD', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4)
        if not logger['complex'] or order == 2:
            axs1.plot(dim_list, non_ana_norm_inj_list, label=f'Order {order} ANA', linewidth=2, color=plt_config.color_dict['order_'+str(order)+'_ana'])
            axs3.plot(dim_list, sym_ana_norm_inj_list, label=f'Order {order} ANA', linewidth=2, color=plt_config.color_dict['order_'+str(order)+'_ana'])
        axs4.plot(dim_list, ratio_list, label=f'Order {order} NGD Ratio', marker='o', linewidth=plt_config.line_width, color=plt_config.color_dict['order_'+str(order)+'_ngd'], markersize=4)
        count += 4
    axs4.plot(dim_list, np.ones(len(dim_list)), linewidth=2, color=plt_config.color_dict['order_2_ana'], label='Order 2 ANA')
    

    axs1.set_xlabel(r'Dimension (\(d\))', fontsize=22)
    axs3.set_xlabel(r'Dimension (\(d\))', fontsize=22)
    axs4.set_xlabel(r'Dimension (\(d\))', fontsize=22)

    axs1.set_ylabel(r'Normalized Injective Norm \(\frac{\|\Psi\|_\varepsilon}{\|\Psi\|}\)', fontsize=22)
    axs3.set_ylabel(r'Normalized Injective Norm \(\frac{\|\Psi\|_\varepsilon}{\|\Psi\|}\)', fontsize=22)
    axs4.set_ylabel(r'Ratio \(\frac{\|\Psi\|_\varepsilon/\|\Psi\|}{(\|\Psi\|_\varepsilon/\|\Psi\|)_\text{sym}}\)', fontsize=22)

    yticks1 = axs1.get_yticks()
    yticks3 = axs3.get_yticks()
    yticks4 = axs4.get_yticks()

    axs1.set_yticklabels(labels=[f"{x:.2f}" for x in yticks1])
    axs3.set_yticklabels(labels=[f"{x:.2f}" for x in yticks3])
    axs4.set_yticklabels(labels=[f"{x:.2f}" for x in yticks4])

    fig1.legend(loc='lower center', bbox_to_anchor=(0.5, 0.86), edgecolor='None', framealpha=0, ncol=count, labelcolor='linecolor', fontsize=16)
    fig3.legend(loc='lower center', bbox_to_anchor=(0.5, 0.86), edgecolor='None', framealpha=0, ncol=count, labelcolor='linecolor', fontsize=16)
    fig4.legend(loc='lower center', bbox_to_anchor=(0.5, 0.86), edgecolor='None', framealpha=0, ncol=count, labelcolor='linecolor', fontsize=16)
    fig1.savefig(f"Plots/NonGauss_Inj_Ratios_{str(order_list)}_cmplx_{logger['complex']}.pdf", bbox_inches='tight')
    fig3.savefig(f"Plots/SymGauss_Inj_Ratios_{str(order_list)}_cmplx_{logger['complex']}.pdf", bbox_inches='tight')
    fig4.savefig(f"Plots/MixGauss_Inj_Ratios_{str(order_list)}_cmplx_{logger['complex']}.pdf", bbox_inches='tight')
    if config.show:
        plt.show()

#endregion


#region

def plot_mps_vs_par_gauss(path1, path2, periodic, rep, cmplx, config=plt_config):
    f = open(path1,)
    logger = json.load(f)
    f.close()

    assert cmplx == True
    assert periodic == True
    assert rep == True
    f = open(path2,)
    gauss_logger = json.load(f)
    f.close()

    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    order_list = [3]
    count = 0
    plt.figure(figsize=(16, 9))
    
    params = utils.mps_fitter(path1, order_list)
    for order in order_list:
        
        for d_idx, dim in enumerate(logger[str(order)]['dim_list']):
            if not dim % 10:
                ngd_normalized_gm_list = []
                for bond_dim in logger[str(order)][str(dim)]['bond_dim_list']:
                    ngd_normalized_gm_list.append(np.mean(np.divide(logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_gm_list'], logger[str(order)][str(dim)][str(bond_dim)]['non_l2_norm_list'])))
                    if dim == 10 and bond_dim == 50:
                        ngd_normalized_gm_list[-1] += 0.0025
                plt.errorbar(logger[str(order)][str(dim)]['bond_dim_list'], ngd_normalized_gm_list, label=f'MPS Order {order}, Dim {dim}', marker='o', linewidth=plt_config.line_width, color=plt_config.mps_col_dict['dim_'+str(dim)], markersize=4, capsize=2)
                plt.axhline(np.mean(np.divide(gauss_logger[str(order)][str(dim)]['par_ngd_gm_list'], gauss_logger[str(order)][str(dim)]['par_l2_norm_list'])), linestyle='solid', linewidth=plt_config.line_width, color=plt_config.mps_col_dict['dim_'+str(dim)], label=f"Guass Order: {order}, Dim: {dim}")
                plt.plot(logger[str(order)][str(dim)]['bond_dim_list'], (params[0]/math.sqrt(dim) + params[1]/np.sqrt(np.array(logger[str(order)][str(dim)]['bond_dim_list'])) + params[2]/np.sqrt(dim*np.array(logger[str(order)][str(dim)]['bond_dim_list'])) + params[3]/dim + params[4]/np.array(logger[str(order)][str(dim)]['bond_dim_list']) + params[5]/(dim*np.array(logger[str(order)][str(dim)]['bond_dim_list'])) )**2, label=f'Order {order}, Dim {dim} EST', color=plt_config.mps_col_dict['dim_'+str(dim)], linestyle='dashed')
            
            count += 1
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=6, edgecolor='None', framealpha=0, fontsize=12, labelcolor='linecolor')
    plt.ylabel(r'Average Normalized Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    plt.xlabel(r'Bond Dimension (\(q\))', fontsize=22)
    plt.savefig(f"Plots/DF_Gauss_MPS_{str(order_list)}_periodic_{str(periodic)}_rep_{str(rep)}_cmplx_{str(cmplx)}_wo.pdf", bbox_inches='tight')
    if config.show:
        plt.show()


def plot_mps_vs_non_gauss(path1, path2, periodic, rep, cmplx, config=plt_config):

    assert periodic == True
    assert rep == False
    assert cmplx == True

    f = open(path1,)
    logger = json.load(f)
    f.close()

    f = open(path2,)
    gauss_logger = json.load(f)
    f.close()

    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    order_list = [3]
    count = 0
    plt.figure(figsize=(16, 9))
    
    params = utils.mps_fitter(path1, order_list)
    for order in order_list:
        
        for d_idx, dim in enumerate(logger[str(order)]['dim_list']):
            if not dim % 10:
                ngd_normalized_gm_list = []
                for bond_dim in logger[str(order)][str(dim)]['bond_dim_list']:
                    ngd_normalized_gm_list.append(np.mean(np.divide(logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_gm_list'], logger[str(order)][str(dim)][str(bond_dim)]['non_l2_norm_list'])))
                    
                plt.errorbar(logger[str(order)][str(dim)]['bond_dim_list'], ngd_normalized_gm_list, label=f'MPS Order {order}, Dim {dim}', marker='o', linewidth=plt_config.line_width, color=plt_config.mps_col_dict['dim_'+str(dim)], markersize=4, capsize=2)
                plt.axhline(np.mean(np.divide(gauss_logger[str(order)][str(dim)]['non_ngd_gm_list'], gauss_logger[str(order)][str(dim)]['non_l2_norm_list'])), linestyle='solid', linewidth=plt_config.line_width, color=plt_config.mps_col_dict['dim_'+str(dim)], label=f"Guass Order: {order}, Dim: {dim}")
                plt.plot(logger[str(order)][str(dim)]['bond_dim_list'], (params[0]/math.sqrt(dim) + params[1]/np.sqrt(np.array(logger[str(order)][str(dim)]['bond_dim_list'])) + params[2]/np.sqrt(dim*np.array(logger[str(order)][str(dim)]['bond_dim_list'])) + params[3]/dim + params[4]/np.array(logger[str(order)][str(dim)]['bond_dim_list']) + params[5]/(dim*np.array(logger[str(order)][str(dim)]['bond_dim_list'])) )**2, label=f'Order {order}, Dim {dim} EST', color=plt_config.mps_col_dict['dim_'+str(dim)], linestyle='dashed')

            count += 1
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=6, edgecolor='None', framealpha=0, fontsize=12, labelcolor='linecolor')
    plt.ylabel(r'Average Normalized Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    plt.xlabel(r'Bond Dimension (\(q\))', fontsize=22)
    plt.savefig(f"Plots/DF_Gauss_MPS_{str(order_list)}_periodic_{str(periodic)}_rep_{str(rep)}_cmplx_{str(cmplx)}_wo.pdf", bbox_inches='tight')
    if config.show:
        plt.show()


def plot_mps_bf(path, periodic, rep, cmplx, config=plt_config, fit=False):

    f = open(path,)
    logger = json.load(f)
    f.close()

    logger = utils.swap_bf_df(logger)

    if config.order_list is None:
        order_list = logger['order_list']
    else:
        order_list = config.order_list

    plt.figure(figsize=(16, 9))
    
    if fit:
        params = utils.mps_fitter(path, order_list)

    for order in order_list:
        
        for b_idx, bond_dim in enumerate(logger[str(order)]['bond_dim_list']):

            # if not bond_dim % 10:  # Uncomment if you want to exclude some points.
            ngd_normalized_gm_list = []
            for dim in logger[str(order)][str(bond_dim)]['dim_list']:
                ngd_normalized_gm_list.append(np.mean(np.divide(logger[str(order)][str(bond_dim)][str(dim)]['non_ngd_gm_list'], logger[str(order)][str(bond_dim)][str(dim)]['non_l2_norm_list'])))

            # Color scheme for the paper.  
            # plt.errorbar(logger[str(order)][str(bond_dim)]['dim_list'], ngd_normalized_gm_list, label=f'MPS Order {order}, Bond Dim {bond_dim}', marker='o', linewidth=plt_config.line_width, color=plt_config.mps_col_dict['dim_'+str(bond_dim)], markersize=4, capsize=2)
            plt.errorbar(logger[str(order)][str(bond_dim)]['dim_list'], ngd_normalized_gm_list, label=f'MPS Order {order}, Bond Dim {bond_dim}', marker='o', linewidth=plt_config.line_width, markersize=4, capsize=2)
            if fit:
                plt.plot(logger[str(order)][str(bond_dim)]['dim_list'], (params[0]/np.sqrt(np.array(logger[str(order)][str(bond_dim)]['dim_list'])) + params[1]/math.sqrt(bond_dim) + params[2]/np.sqrt(bond_dim*np.array(logger[str(order)][str(bond_dim)]['dim_list'])) + params[3]/np.array(logger[str(order)][str(bond_dim)]['dim_list'])  +  params[4]/bond_dim + params[5]/(bond_dim*np.array(logger[str(order)][str(bond_dim)]['dim_list'])) )**2, label=f'Order {order}, Dim {dim} EST', color=plt_config.mps_col_dict['dim_'+str(bond_dim)], linestyle='dashed')

    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=6, edgecolor='None', framealpha=0, fontsize=12, labelcolor='linecolor')
    plt.ylabel(r'Average Normalized Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    plt.xlabel(r'Physical Dimension (\(d\))', fontsize=22)
    plt.savefig(f"Plots/BF_Gauss_MPS_{str(order_list)}_periodic_{str(periodic)}_rep_{str(rep)}_cmplx_{str(cmplx)}_wo.pdf", bbox_inches='tight')
    if config.show:
        plt.show()

#endregion


# region Deterministic States

def plot_dicke(path, config=plt_config):


    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()

    num_particles = np.sum(logger['partition_tuple_list'][0])

    cmap = plt.get_cmap(config.cmap)
    plt.figure(figsize=(16, 9))

    plt.plot(range(len(logger['partition_tuple_list'][1:])), logger['ana_gm_list'][1:], color='orange', marker='o', markersize=20, label='ANA', linewidth=0)
    plt.plot(range(len(logger['partition_tuple_list'][1:])), np.array(logger['als_gm_list'][1:])**2, color='green', marker='+', markersize=20, label='ALS', linewidth=0)
    plt.plot(range(len(logger['partition_tuple_list'][1:])), np.array(logger['ngd_gm_list'][1:])**2, color='blue', marker='x', markersize=16, label='NGD', linewidth=0)

    plt.xticks(range(len(logger['partition_tuple_list'][1:])), labels=[str(list(x))for x in logger['partition_tuple_list'][1:]]) 

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=5, edgecolor='None', framealpha=0, fontsize=16, labelcolor='linecolor')
    plt.ylabel(r'Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    plt.xlabel(r'Partition tuple', fontsize=22)
    plt.savefig(f"Plots/Dicke_{str(num_particles)}.pdf", bbox_inches='tight')
    if config.show:
        plt.show()


def plot_antisym(path, config=plt_config):


    # Load the dictionary
    f = open(path,)
    logger = json.load(f)
    f.close()

    cmap = plt.get_cmap(config.cmap)
    plt.figure(figsize=(16, 9))

    plt.plot(logger['dim_list'], logger['ana_gm_list'], color='orange', marker='o', markersize=20, label='ANA', linewidth=0)
    plt.plot(logger['dim_list'], [x**2 for x in logger['als_gm_list']], color='green', marker='+', markersize=20, label='ALS', linewidth=0)
    plt.plot(logger['dim_list'], [x**2 for x in logger['ngd_gm_list']], color='blue', marker='x', markersize=16, label='NGD', linewidth=0)


    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=5, edgecolor='None', framealpha=0, fontsize=16, labelcolor='linecolor')
    plt.ylabel(r'Injective Norm \(\|\Psi\|_\varepsilon\)', fontsize=22)
    plt.xlabel(r'Dimension (\(d\)) = Order (\(n\))', fontsize=22)
    plt.savefig(f"Plots/Antisym.pdf", bbox_inches='tight')
    if config.show:
        plt.show()  


# endregion