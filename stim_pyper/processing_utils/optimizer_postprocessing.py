import os
import json
from stim_pyper.vta_model.model_directional_vta import ModelDirectionalVta

def process_vta(amplitudes, coordinates, side, output_dir, voxel_size=[0.2, 0.2, 0.2], grid_shape=[71, 71, 71]):
    filenames = []
    stimparams = os.path.join(output_dir, f'stimparams_elec_{side}.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(stimparams, 'w') as json_file:
        json.dump(amplitudes.tolist(), json_file)

    for amplitude, coord in zip(amplitudes, coordinates):
        if amplitude == 0:
            continue
        coord_tuple = tuple(coord)
        index = [tuple(c) for c in coordinates].index(coord_tuple)
        updated_fname = f'single_contact_elec_{side}_contact_{index}.nii'
        coord = coord.tolist()
        group_indices = []
        if index in [1, 2, 3]:
            group_indices = [1, 2, 3]
        elif index in [4, 5, 6]:
            group_indices = [4, 5, 6]

        contact_coordinates = [coord] + [coordinates[i].tolist() for i in group_indices if i != index]
        vta = ModelDirectionalVta(contact_coordinates=[contact_coordinates[0]],
            voxel_size=voxel_size,
            grid_shape=grid_shape,
            output_path=output_dir,
            fname=updated_fname)
        vta.run(radius_mm=amplitude)
        filenames.append(vta.fpath)
    return filenames

