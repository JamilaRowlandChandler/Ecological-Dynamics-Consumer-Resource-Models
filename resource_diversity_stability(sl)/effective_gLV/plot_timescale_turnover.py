# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:25:53 2026

@author: jamil
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
import seaborn as sns

# %%

def divergence_dist_time(data0,
                         data1 = None):
    
    def reshape_data(data):
        
        return data.reshape((data.shape[0],
                             data.shape[1] * data.shape[2]))
    
    def dist_within_data(data):
        
        sorted_data = np.sort(data, axis=1)
        pairwise_dists = squareform(pdist(sorted_data,
                                          metric="cityblock")) / data.shape[1]
        
        diagonal_mask = np.eye(dist_t0.shape[0],
                               dtype=bool)
         
        mean_dist = np.mean(pairwise_dists[~diagonal_mask])
        
        return mean_dist
    
    def dist_between_data(data0, data1):
        
        sorted_data0 = np.sort(data0, axis=1)
        sorted_data1 = np.sort(data1, axis=1)
        pairwise_dists = cdist(sorted_data0,
                               sorted_data1,
                               metric="cityblock") / data0.shape[1]
        
        mean_dist = np.mean(pairwise_dists)
        
        return mean_dist
        
    dist_t0 = reshape_data(data0)
    
    if data1 is not None:
        
        dist_t1 = reshape_data(data1)
        
        mean_dist = dist_between_data(dist_t0, dist_t1)
        
    else:
        
        mean_dist = dist_within_data(dist_t0)
    
    return mean_dist

# %%

def plots(data,
          filename):
    
    def negative_interaction_count(A_data):
        
        reshaped_A = A_data.reshape(A_data.shape[0],
                                    A_data.shape[1] * A_data.shape[2])
        
        any_neg = np.any(reshaped_A < 0, axis = 1)
        all_neg = np.all(reshaped_A < 0, axis = 1)

        any_all_neg = np.sum(any_neg == all_neg)/len(any_neg.flatten())
        
        print(np.sum(any_neg), "\n",
              np.sum(all_neg), "\n",
              np.sum(all_neg)/np.sum(any_neg), "\n",
              any_all_neg, "\n")
        
    def cv_plot():
        
        cv001 = A_data_001.std(axis=(1, 2), ddof=1) / np.abs(A_data_001.mean(axis=(1, 2)))
        cv1 = A_data_1.std(axis=(1, 2), ddof=1) / np.abs(A_data_1.mean(axis=(1, 2)))

        fig, (ax1, ax2) = plt.subplots(2, 1,
                                       layout="constrained",
                                       sharex=True)

        ax1.hist(cv001,
                30,
                fill=None,
                label=label_001,
                edgecolor='red',
                density=True)
         #)
        ax1.legend()
            
        ax2.hist(cv1,
                 50,
                 fill=None,
                 label=label_1,
                 edgecolor='blue',
                 density=True)
           #)
        ax2.legend()
        
        fig.supxlabel("effective consumer interactions, " + \
                      r'$A_{ij}$')
        fig.supylabel("density")
        
        #plt.savefig()
        
        plt.show()
        
    def interactions_all_time_plot():
        
        fig, ax = plt.subplots(1, 1, layout="constrained",
                               figsize=(4,3))
        
        data_plot = pd.concat([pd.DataFrame({'epsilon' : label_001,
                                  'A' : A_data_001.flatten()}),
                                 pd.DataFrame({'epsilon' : label_1,
                                               'A' : A_data_1.flatten()})])
             
        sns.histplot(data_plot,
                     x = 'A', hue = 'epsilon',
                     binwidth=10,
                     stat='density',
                     ax=ax,
                     legend=False)
        plt.xlim(-1000, 1000)
        plt.legend(title='timescale difference',
                   loc='upper left',
                   labels=[label_1, label_001])
        
        ax.set_xlabel("effective consumer interactions, " + \
                      r'$A_{ij}$')
            
        plt.savefig(figure_directory + "/A_all_time_" + filename + ".png",
                    bbox_inches='tight')
        plt.savefig(figure_directory + "/A_all_time_" + filename + ".svg",
                    bbox_inches='tight')
            
    def interactions_per_time_plot(indexes):
        
        fig, axs = plt.subplots(2, int(np.ceil(len(indexes)/2)),
                                layout="constrained",
                                figsize=(2.5*len(indexes),6),
                                sharex=True,
                                sharey=True)
        
        for ax, idx in zip(axs.flatten(), indexes):
        
            data_plot = pd.concat([pd.DataFrame({'epsilon' : label_001,
                                      'A' : A_data_001[idx, :, :].flatten()}),
                                     pd.DataFrame({'epsilon' : label_1,
                                                   'A' : A_data_1[idx, :, :].flatten()})])
             
            sns.histplot(data_plot,
                         x = 'A', hue = 'epsilon',
                         binwidth=10,
                         stat='density',
                         ax=ax,
                         legend=False)
            plt.xlim(-1000, 1000)
            
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("t = " + str(7000 + (float(idx) * 7000.0/200.0)))
                
        fig.supxlabel("effective consumer interactions, " + \
                      r'$A_{ij}$')
        fig.supylabel("density")
        
        plt.legend(title='timescale difference',
                   loc='upper left',
                   labels=[label_1, label_001])
        
        plt.savefig(figure_directory + "/A_over_time_" + filename + ".png",
                    bbox_inches='tight')
        plt.savefig(figure_directory + "/A_over_time_" + filename + ".svg",
                    bbox_inches='tight')
        
        plt.show()
    
    A_data_001 = data['A'][0][50:, :, :]
    A_data_1 = data['A'][1][50:, :, :]
    label_001 = str(np.round(1/data['epsilon'][0], 4))
    label_1 = str(np.round(1/data['epsilon'][1], 4))
    
    #negative_interaction_count(A_data_1)
    #negative_interaction_count(A_data_001)
    
    #divergence_dist_times = [divergence_dist_time(A_data_1),
    #                         divergence_dist_time(A_data_001),
    #                         divergence_dist_time(A_data_1,
    #                                              A_data_001)]
    
    #cv_plot()
    interactions_all_time_plot()
    interactions_per_time_plot(np.arange(-5, 0, 1))

# %%

M_stab_stats = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data/resource_diversity_stability/simulations/" + \
                              "CRM_TS/properties_over_time" + \
                              "/timescalar_2_1_001_example_interactions.pkl")
figure_directory = "C:/Users/jamil/Documents/PhD/Figures/resource_diversity_stability"

# %%
                    
plots(M_stab_stats['50']['unstable'],
      "M_50_unstable")

plots(M_stab_stats['250']['stable'],
      "M_250_stable")

# %%


# %%


