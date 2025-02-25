import utils
import matplotlib.pyplot as plt

class def_config():
    seed = 91219
    initialization=None
    lr=1e-3
    optimizier='ADAM'
    criterion=utils.my_criterion
    max_epoch=10000
    thresh_loss=1e-9
    initial_patience=10
    saturation_fac=1e-9
    patience_perc=5
    normalize=True
    leave=True
    ret_loss=True
    decomp_only_init=False
    rtol = 3e-3

cmap = plt.get_cmap('tab10')

class plt_config():
    dim_list = None #range(250, 25, -5)
    order_list = None
    axis_fz = 12
    head_fz = 16
    marker_list = ['x', '+', 'o', 'D']
    lin_alpha = 2.5
    show = False
    line_alpha = 1
    line_width = 2
    color_dict = {
        'order_3_fit': 'limegreen',
        'order_2_fit': 'goldenrod',
        'order_2_ana': 'palevioletred',
        'order_3_ana': 'purple',
        'order_2_als': cmap(1),
        'order_3_als': cmap(3),
        'order_2_ngd': 'dodgerblue',
        'order_3_ngd': 'navy',
        'order_2_sgd': cmap(5),
        'order_3_sgd': cmap(7),
        'order_2_pow': cmap(8),
        'order_3_pow': cmap(9)
        }
    mps_col_dict = {
        'dim_10': 'limegreen',
        'dim_20': 'goldenrod',
        'dim_30': 'palevioletred',
        'dim_40': 'purple',
        'dim_50': 'dodgerblue',
        'dim_60': 'navy'
    }
        
    '''
        'cyan' 
        'royal'
        'blue'
        'crimson'
        'red'
        'lime'
        'fuchsia'
        'dark'
        'violet'
        'lawngreen'
    '''

