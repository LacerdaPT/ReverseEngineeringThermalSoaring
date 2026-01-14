import numpy as np

xlabel_dict = {'x': '$\\delta v_{x}\\ \\ (\\mathrm{{m~s}}^{{-1}})$',
               'y': '$\\delta v_{y}\\ \\ (\\mathrm{{m~s}}^{{-1}})$',
               'z': '$\\delta v_{z}\\ \\ (\\mathrm{{m~s}}^{{-1}})$',
               'xyz': '$\\lvert \\lvert  \\delta \\boldsymbol{v} \\rvert \\rvert\\ \\ (\\mathrm{{m~s}}^{{-1}})$'}

component_to_label_mapping = {'inner': '$C\\left( \\Delta R \\right)$',
                              'x': '$C_x\\left( \\Delta R \\right)$',
                              'y': '$C_y\\left( \\Delta R \\right)$',
                              'z': '$C_z\\left( \\Delta R \\right)$'}
#
def plot_fluctuations_histogram(ax_all,df_plot_fluctuations, list_of_components, partition_pair,
                                do_gt=True,df_all_sigma_grouped = None, n_sigmas=3, **kwargs):
    list_of_datatypes = ['dec', 'gt'][::-1] if do_gt else ['dec']
    partition_key, partition_values = partition_pair
    #ax_all = np.array(list(ax_all.values())).reshape(2,2)
    for i_partition, partition_val in enumerate(partition_values):
        #ax_row = ax_all[i_radius]
        current_fluctuations_partition = df_plot_fluctuations[df_plot_fluctuations[partition_key] == partition_val].copy()

        for i_comp, comp in enumerate(list_of_components):
            for i_datatype, datatype in enumerate(list_of_datatypes):
                current_fluctuations = current_fluctuations_partition.loc[
                    current_fluctuations_partition['datatype'] == datatype, comp]
                if df_all_sigma_grouped is not None:
                    current_sigma = df_all_sigma_grouped.loc[df_all_sigma_grouped[partition_key] == partition_val, f'{comp}_mean'].max()
                    current_fluctuations = current_fluctuations[np.abs(current_fluctuations) < n_sigmas * current_sigma]
                ax_all[i_comp, i_partition].hist(current_fluctuations,
                                              bins=int(np.sqrt(len(current_fluctuations))), label=datatype.upper(),**kwargs)
            # ax_all[i_comp, i_partition].set_xlabel(f'$\\delta v_{{{comp}}} \\ (ms^{{-1}})$')
