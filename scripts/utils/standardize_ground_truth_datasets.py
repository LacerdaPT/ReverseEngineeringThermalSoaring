import os
from pathlib import Path


from data.get_data import load_synthetic_data
from data.synthetic.post import postprocess_generated_datasets

save = True
overwrite = True
root_path = '/home/pedro/PycharmProjects/ReverseEngineeringThermalSoaring'

base_path = 'migration_test'
path_wildcard = 'None'
list_of_bad = []
for ss in Path(os.path.join(root_path, base_path)).glob(path_wildcard):
    try:

        path_to_synthetic = str(ss)
        path_to_save = os.path.join(path_to_synthetic)
        if os.path.exists(os.path.join(path_to_save, 'bird.csv')) and (not overwrite):
            continue

        print(path_to_synthetic)
        if save:
            os.makedirs(path_to_save, exist_ok=True)

        syn = load_synthetic_data(path_to_synthetic, list_of_object=['data_real.csv','data.csv'])
        df_data = syn['data']
        df_real = syn['data_real']
        df_data_full, df_air, df_bird = postprocess_generated_datasets(df_real, df_data)
        if save:
            df_data_full.to_csv(os.path.join(path_to_save, 'data_full.csv'), index=False)
            df_air.to_csv(os.path.join(path_to_save, 'air.csv'), index = False)
            df_bird.to_csv(os.path.join(path_to_save, 'bird.csv'), index = False)
        print('done')
    except Exception as e:
        list_of_bad.append(path_to_synthetic)
        print('SKIPPING ', path_to_synthetic)