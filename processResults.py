#%%
import os
from os import walk, sep
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#import sys
#sys.path.append('./')

# %%
# process predictive results

def concat_results():
    _, _, result_files = next(walk(f'results_modeling{sep}test{sep}'))
    concat_results = pd.DataFrame()
    for _, file in enumerate(result_files):
        results = pd.read_csv(
            f'results_modeling{sep}test{sep}{file}')
        results['orig_oracle_acc'] = None
        results['orig_oracle_f1'] = None
        results['orig_oracle_auc'] = None
        if "orig" in file:
            results_cv_orig = pd.read_csv(
            f'results_modeling{sep}validation{sep}{file}')
            print()
            # guaranteeing that we have the best model in the grid search instead of 3 models
            result_cv_acc = results.iloc[[results_cv_orig['mean_test_bal_acc'].idxmax()]]
            result_cv_f1 = results.iloc[[results_cv_orig['mean_test_f1_weighted'].idxmax()]]
            result_cv_auc = results.iloc[[results_cv_orig['mean_test_roc_auc_weighted'].idxmax()]]

            results['orig_oracle_acc'] = result_cv_acc.test_balanced_accuracy.values[0]
            results['orig_oracle_f1'] = result_cv_f1.test_f1_weighted.values[0]
            results['orig_oracle_auc'] = result_cv_auc.test_roc_auc_balanced.values[0]
                
        # get dataset metadata
        results['technique'] = file.split('.csv')[0] 
        
        concat_results = results if concat_results.empty else pd.concat(
            [concat_results, results])

    return concat_results


# %% AGNOSTIC
results = concat_results()
# %%
orig = results.loc[results.technique=='orig'].reset_index(drop=True)

# %%
results = results.loc[results.technique!='orig'].reset_index(drop=True)
del results['orig_oracle_acc']
del results['orig_oracle_f1']
del results['orig_oracle_auc']
# %%
# each model against the respective model in original data
results['test_balanced_accuracy_perdiff'] = None
results['test_roc_auc_balanced_perdiff'] = None
results['test_f1_weighted_perdiff'] = None

# each model against the best model in original data
results['test_balanced_accuracy_perdiff_oracle'] = None
results['test_roc_auc_balanced_perdiff_oracle'] = None
results['test_f1_weighted_perdiff_oracle'] = None

for i in range(len(results)):
    orig_model = orig.loc[orig.params == results.params[i],:].reset_index()
    results['test_balanced_accuracy_perdiff'][i] = 100*(results['test_balanced_accuracy'][i]-orig_model['test_balanced_accuracy'][0]) / orig_model['test_balanced_accuracy'][0]
    results['test_roc_auc_balanced_perdiff'][i] = 100*(results['test_f1_weighted'][i]-orig_model['test_f1_weighted'][0]) / orig_model['test_f1_weighted'][0]
    results['test_f1_weighted_perdiff'][i] = 100*(results['test_roc_auc_balanced'][i]-orig_model['test_roc_auc_balanced'][0]) / orig_model['test_roc_auc_balanced'][0]

    results['test_balanced_accuracy_perdiff_oracle'][i] = 100*(results['test_balanced_accuracy'][i]-orig_model['orig_oracle_acc'][0]) / orig_model['orig_oracle_acc'][0]
    results['test_roc_auc_balanced_perdiff_oracle'][i] = 100*(results['test_f1_weighted'][i]-orig_model['orig_oracle_f1'][0]) / orig_model['orig_oracle_f1'][0]
    results['test_f1_weighted_perdiff_oracle'][i] = 100*(results['test_roc_auc_balanced'][i]-orig_model['orig_oracle_auc'][0]) / orig_model['orig_oracle_auc'][0]

# %%
columns = ['test_balanced_accuracy_perdiff', 'test_roc_auc_balanced_perdiff', 'test_f1_weighted_perdiff']

# Melt the DataFrame to long format for seaborn
results_melted = results.melt(id_vars=['model','technique'], value_vars=columns,
                    var_name='perdif', value_name='value')


# %%
results_melted.loc[results_melted['model'].str.contains('SVC'), 'model'] = 'SVM'
# Dummy data for illustration
categories = ['Random Forest', 'Bagging', 'XGBoost', 'SVM']

metrics=['Accuracy', 'AUC', 'F-score']

# %%
sns.set_style("darkgrid")
# Initialize the figure and axes
fig, axes = plt.subplots(2, 3, figsize=(14, 12))
colors=['#03045e', "#0077b6","#00b4d8"]

results_melted_sup = results_melted.loc[results_melted.technique=='sup']
sns.barplot(data=results_melted_sup, x='model', y='value',hue='perdif',
            ax=axes[0,0],legend=False,palette=colors)
axes[0,0].set_ylabel('Percentage difference \n of predictive performance (F-score)')
axes[0,0].set_xlabel('')
axes[0,0].set_title('Suppression (' +r"$\bf{" + str(100) + "\%}$"+')')
axes[0,0].set_xticklabels('')
axes[0,0].set_yticks(np.arange(-20, 35, 5))

results_melted_gen = results_melted.loc[results_melted.technique=='gen']
sns.barplot(data=results_melted_gen, x='model', y='value',hue='perdif',
            ax=axes[0,1],palette=colors)
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel('')
axes[0,1].set_xticklabels('')
axes[0,1].set_title('Generalisation (' +r"$\bf{" + str(100) + "\%}$"+')')
axes[0,1].set_yticks(np.arange(-20, 35, 5))
handles,labels = axes[0,1].get_legend_handles_labels()
labels=metrics
axes[0,1].legend(handles,labels)
sns.move_legend(axes[0,1], bbox_to_anchor=(0.5,1.2), loc='upper center', borderaxespad=0., ncol=3, frameon=False, title='')

results_melted_supgen = results_melted.loc[results_melted.technique=='supgen']
sns.barplot(data=results_melted_supgen, x='model', y='value',hue='perdif',
            ax=axes[0,2],legend=False,palette=colors)
axes[0,2].set_ylabel('')
axes[0,2].set_xlabel('')
axes[0,2].set_xticklabels('')
axes[0,2].set_title('Suppression and \ngeneralisation (' +r"$\bf{" + str(100) + "\%}$"+')')
axes[0,2].set_yticks(np.arange(-20, 35, 5))

results_melted_supnoi = results_melted.loc[results_melted.technique=='supnoi']
sns.barplot(data=results_melted_supnoi, x='model', y='value',hue='perdif',
            ax=axes[1,0],legend=False,palette=colors)
axes[1,0].set_ylabel('')
axes[1,0].set_xlabel('')
axes[1,0].tick_params(axis='x', labelrotation=30)
axes[1,0].set_title('Suppression and noise (' +r"$\bf{" + str(71.23) + "\%}$"+')')
axes[1,0].set_yticks(np.arange(-20, 35, 5))
axes[1,0].set_ylabel('Percentage difference \n of predictive performance (F-score)')

results_melted_gennoi = results_melted.loc[results_melted.technique=='gennoi']
sns.barplot(data=results_melted_gennoi, x='model', y='value',hue='perdif',
            ax=axes[1,1],legend=False,palette=colors)
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel('')
axes[1,1].tick_params(axis='x', labelrotation=30)
axes[1,1].set_title('Generalisation and \nnoise (' +r"$\bf{" + str(35.05) + "\%}$"+')')
axes[1,1].set_yticks(np.arange(-20, 35, 5))

results_melted_supgennoi = results_melted.loc[results_melted.technique=='supgennoi']
sns.barplot(data=results_melted_supgennoi, x='model', y='value',hue='perdif',
            ax=axes[1,2],legend=False,palette=colors)
axes[1,2].set_ylabel('')
axes[1,2].set_xlabel('')
axes[1,2].tick_params(axis='x', labelrotation=30)
axes[1,2].set_title('Suppression, generalisation \nand noise (' +r"$\bf{" + str(0) + "\%}$"+')')
axes[1,2].set_yticks(np.arange(-20, 35, 5))

sns.set(font_scale=1.2)
fig.savefig(f'Plots/all_results.pdf', bbox_inches='tight')


# %% boxplot of raw values
all_results = concat_results()
all_results.loc[all_results['model'].str.contains('SVC'), 'model'] = 'SVM'
#%%
sns.set_style("darkgrid")
colors=['#e76f51','#f4a261','#e9c46a','#fff3b0']
g = sns.FacetGrid(all_results, col="technique", col_order=['orig','sup','gen','supgen','supnoi','gennoi','supgennoi'],
                  hue='model',height=3.5, aspect=0.56,gridspec_kws={"wspace":0.05},palette=colors)
g.map(sns.boxplot,"model", "test_f1_weighted")
g.set_xticklabels('')
g.set_xlabels('')
g.set_ylabels('F-score')
g.set(yticks = np.arange(0.35, 0.75, 0.05))
#g.set_yticks(np.arange(30, 70, 5))
# flatten the array of axes for easy iteration and usage
axes = g.axes.flat

title=['Baseline', 'Suppression', 'Generalisation',
       'Suppression and \nGeneralisation', 'Suppression \nand noise',
       'Generalisation \nand noise',
       'Suppression, \nGeneralisation \nand noise']
# rotate the titles by iterating through each axes
for ax,i in zip(axes,range(7)):
    #title = ax.get_title()
    #print(title)
    ax.set_title(title[i])
g.add_legend()
sns.move_legend(g,bbox_to_anchor=(0.5,0), loc='lower center', borderaxespad=0., ncol=4, frameon=False, title='')

sns.set(font_scale=1)
g.fig.tight_layout()
fig.savefig(f'Plots/all_results_boxplot.pdf', bbox_inches='tight')

