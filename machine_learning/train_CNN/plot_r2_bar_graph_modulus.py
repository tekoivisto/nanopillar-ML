# Plot bar graph and scatter plot of the CNN and averaging method predictions
# for the Young's modulus

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.transforms as transforms
from matplotlib.gridspec import GridSpec

def argmedian(arr):
    sorted_indices = np.argsort(arr)
    n = len(arr)
    mid_index = (n - 1) // 2
    return sorted_indices[mid_index]

def r_2(y_true, y_pred):
    RSS =  np.sum(np.square(y_true - y_pred))
    TSS = np.sum(np.square(y_true - np.mean(y_true)))
    return ( 1 - (RSS/TSS) )

plt.style.use('seaborn-v0_8-paper')
sns.set_context('paper')

gt = 'modulus'

gt_dir = 'moduluses_0.0015'
gt_name = 'Young\'s modulus'
scatter_tick_pos = [150, 170, 190]
legend_pos = (0.13, 0)
fig_width=8.5
scatter_title='(c)'
lim_pad = 4

y_gt = np.load(f'../ground_truth/test_set/{gt_dir}.npy')
y_gt_tr_val_set = np.load(f'../ground_truth/tr_val_set/{gt_dir}.npy')

color_plot = [
    'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue',  'tab:orange', 'tab:orange'
]

dir_tree_engineered = [[gt_dir],
            ['grain_boundary', 'quaternion', 'combined'],
            ['16','32'],
            ]

dir_tree_raw = [[gt_dir],
            ['raw'],
            ['64', '128'],
            ]

n_seeds=5

def scores_in_dir(path):
    n_stats_files_found = 0
    test_scores = []
    train_scores = []

    for file in sorted(os.listdir(path)):
        if file[:5] == 'stats':

            with open(f'{path}/{file}', 'r') as f:
                lines = f.readlines()

                n_stats_files_found += 1
                try:
                    test_score = lines[3].split()[2]
                    train_score = lines[3].split()[0]
                except IndexError:
                    test_score = 0
                    train_score = 0
                    print('error in stats file!', path, n_stats_files_found)
                
                test_scores.append(float(test_score))
                train_scores.append(float(train_score))

    if n_stats_files_found != n_seeds:
        print('all stats files not found!', path)

    return test_scores, train_scores

gt_dict = {'modulus':'Young\'s\nmodulus',
           'yield_stress':'yield stress'}
leg_dict = {'combined':'grain boundary +\nlattice orientation',
            'quaternion':'lattice orientation',
            'grain_boundary':'grain boundary',
            'raw':'atom positions',
            
            'voigt': 'Voigt',
            'reus': 'Reuss',
            'hill': 'Hill',
            'volume integral': 'volume integral'}

label_dict = {'combined':'grain boudary\n    + lattice\n  orientation',
            'quaternion':'    lattice\norientation',
            'grain_boundary':'    grain\nboundary',
            'raw':'   atom\npositions',
            
            'voigt': 'Voigt',
            'reus': 'Reuss',
            'hill': 'Hill',
            'volume integral': 'volume\nintegral'}

alphas= {'16':0.6, '32':1, '64':0.6, '128':1}

scatter_lim = [y_gt.min()-lim_pad, y_gt.max()+lim_pad]

fig = plt.figure()
if gt=='modulus':
    gs = GridSpec(5, 7, figure=fig, width_ratios=[0.5, 0.7, 2.9, 0.7, 1, 1.2, 1], height_ratios=[1,1,0.6,0.4,1])
    ax_bar_zoom = fig.add_subplot(gs[:-2, 2])
else:
    gs = GridSpec(5, 4, figure=fig, width_ratios=[2, 0.5, 1, 1], height_ratios=[1,1,0.6,0.4,1])

ax_bar = fig.add_subplot(gs[:-2, 0])

axs_scatter = {'grain_boundary': fig.add_subplot(gs[0, -3]),
               'quaternion': fig.add_subplot(gs[1, -3]),
               'combined': fig.add_subplot(gs[2:4, -3]),
               'raw': fig.add_subplot(gs[4, -3]),

               'voigt': fig.add_subplot(gs[0, -1]),
               'reus': fig.add_subplot(gs[1, -1]),
               'hill': fig.add_subplot(gs[2:4, -1]),
               'volume integral' : fig.add_subplot(gs[4, -1])
               }


fig.set_size_inches(fig_width,4.5)
fig.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)

gap = 0.5  # Define the gap width between different descriptors
width = 1  # Bar width

resolution_labels = {}  # Store labels for legend
tick_labels = []  # Collecting labels for the x-axis
positions = []  # Collecting positions for the x-axis labels

i=0
x=0
for dir_tree in [dir_tree_engineered, dir_tree_raw]:

    for gt_dir in dir_tree[0]:
        
        for descriptor in dir_tree[1]:
            descriptor_label_added = False
            for res in dir_tree[2]:

                path = f'{gt_dir}/{descriptor}/{res}/'
                if dir_tree == dir_tree_engineered:
                    path += '0.0005/'

                test_scores_dir, train_scores_dir = scores_in_dir(path)

                avg_test_score = np.mean(test_scores_dir)
                std_test_score = np.std(test_scores_dir)
                
                label = f'{leg_dict[descriptor]} ({res})'

                bar = ax_bar.bar(x, avg_test_score, width=width, color=color_plot[i], alpha=alphas[res], label=label)
                ax_bar.errorbar(x, avg_test_score, yerr=std_test_score, ecolor='black', capsize=5, elinewidth=1, markeredgewidth=1)

                if gt=='modulus':
                    ax_bar_zoom.bar(x, avg_test_score, width=width, color=color_plot[i], alpha=alphas[res], label=label)
                    ax_bar_zoom.errorbar(x, avg_test_score, yerr=std_test_score, ecolor='black', capsize=5, elinewidth=1, markeredgewidth=1)

                median_idx = argmedian(test_scores_dir)
                y_pred = np.load(f'{path}/preds_test_set_augmented_{median_idx+1}.npy')
                if res == '32' or res=='128':
                    ax_scatter = axs_scatter[descriptor]
                    ax_scatter.scatter(y_gt, y_pred, s=0.1)
                    ax_scatter.plot(scatter_lim, scatter_lim, 'k--', linewidth=0.7)

                    ax2 = ax_scatter.twinx()
                    ax2.set_ylim([0, 1])
                    ax2.set_yticks([0.5])
                    ax2.set_yticklabels([label_dict[descriptor]], fontsize=10, rotation=90, verticalalignment='center')
                    #ax2.set_yticks(rotation=90)
                    ax2.tick_params(axis='y', length=0)

                    if descriptor != 'raw':
                        ax_scatter.set_xticklabels([])
                    ax_scatter.set_xlim(scatter_lim)
                    ax_scatter.set_ylim(scatter_lim)
                    ax_scatter.set_xticks(scatter_tick_pos)
                    ax_scatter.set_yticks(scatter_tick_pos)
                    ax_scatter.tick_params(axis='both', direction='in', length=4)

                    ax_scatter.text(0.1, 0.95, fr'$r^2$={test_scores_dir[median_idx]:.3f}', transform=ax_scatter.transAxes, verticalalignment='top', fontsize=9)
                resolution_labels[res] = bar

                if not descriptor_label_added:
                    positions.append(x + width / 2)
                    tick_labels.append(leg_dict[descriptor])
                    descriptor_label_added = True

                i+=1

                x += width 

            x += gap



y_direct_methods = np.load('../../averaging_methods/averaging_local_modulus_field/y_preds_direct.npz')
y_direct_methods_tr_val = np.load('../../averaging_methods/averaging_local_modulus_field/y_preds_direct_tr_val.npz')

diret_keys = ['voigt', 'reus', 'hill', 'volume integral']
direct_res = [16, 32]
for key in diret_keys:
    for res in direct_res:

        if key in ['voigt', 'hill']:
            proper_path = '../../averaging_methods/averaging_stiffness_tensor'
            y_pred_data = np.load(f'{proper_path}/{key}_wide_search_{res}.npz')
            y_pred = y_pred_data['E_pred_test_set']
            y_pred_tr_val = y_pred_data['E_pred_tr_val_set']

        else:
            y_pred = y_direct_methods[f'{key} {res}']
            y_pred_tr_val = y_direct_methods_tr_val[f'{key} {res}']
        
        y_pred = (y_pred-np.mean(y_pred_tr_val))/np.std(y_pred_tr_val) * np.std(y_gt_tr_val_set)+np.mean(y_gt_tr_val_set)

        r2 = r_2(y_gt, y_pred)
        ax_bar_zoom.bar(x, r2, width=width, color=color_plot[0], alpha=alphas[str(res)], label=label)

        if res==16:
            positions.append(x + width / 2)
            tick_labels.append(leg_dict[key])
        
        if res == 32:
            ax_scatter = axs_scatter[key]
            ax_scatter.scatter(y_gt, y_pred, s=0.1)
            ax_scatter.plot(scatter_lim, scatter_lim, 'k--', linewidth=0.7)


            ax2 = ax_scatter.twinx()
            ax2.set_ylim([0, 1])
            ax2.set_yticks([0.5])
            ax2.set_yticklabels([label_dict[key]], fontsize=10, rotation=90, verticalalignment='center')
            #ax2.set_yticks(rotation=90)
            ax2.tick_params(axis='y', length=0)

            if key != 'volume integral':
                ax_scatter.set_xticklabels([])
            ax_scatter.set_xlim(scatter_lim)
            ax_scatter.set_ylim(scatter_lim)
            ax_scatter.set_xticks(scatter_tick_pos)
            ax_scatter.set_yticks(scatter_tick_pos)
            ax_scatter.tick_params(axis='both', direction='in', length=4)

            ax_scatter.text(0.1, 0.95, fr'$r^2$={r2:.3f}', transform=ax_scatter.transAxes, verticalalignment='top', fontsize=9)


        x += width

    x += gap

axs_tick = [ax_bar, ax_bar_zoom]
for ax in axs_tick:
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    offset = transforms.ScaledTranslation(5/72, 0, plt.gcf().dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)

handles, labels = [], []
for res, bar in resolution_labels.items():
    handles.append(bar[0])  # Get the bar object
    labels.append(res+r'$\times$'+res+r'$\times$'+f'{2*int(res)}')

fig.legend(handles, labels, title='resolution', loc='lower left', ncol=2, bbox_to_anchor=legend_pos)

ax_bar.set_yticks(np.linspace(0,1, 11), minor=True)
ax_bar.yaxis.grid(True)
ax_bar.yaxis.grid(True, which='minor')
ax_bar.set_ylim([0, 1])
ax_bar.set_ylabel(r'$r^2$')
ax_bar.set_title('(a)')
ax_bar.set_xlim([-1, 2])

ax_bar_zoom.set_xlim([1.5, 19.5])
ax_bar_zoom.set_ylim([0.92, 1])
ax_bar_zoom.set_title('(b)')
ax_bar_zoom.set_ylabel(r'$r^2$')
ax_bar_zoom.yaxis.grid(True)
ax_bar_zoom.yaxis.grid(True, which='minor')

axs_scatter['raw'].set_xlabel(f'true {gt_name} (GPa)')
axs_scatter['combined'].set_ylabel(f'                        predicted {gt_name} (GPa)')
axs_scatter['grain_boundary'].set_title(f'{scatter_title}')
axs_scatter['volume integral'].set_xlabel(f'true {gt_name} (GPa)')
axs_scatter['hill'].set_ylabel(f'                        predicted {gt_name} (GPa)')
axs_scatter['voigt'].set_title(f'(d)')

fig.savefig(f'bar_graph_{gt}.pdf', bbox_inches='tight')

plt.show()
