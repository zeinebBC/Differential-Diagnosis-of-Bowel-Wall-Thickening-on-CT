
def plan_experiment(self):
    """
    MOVE EVERYTHING INTO THE PLANS. MAXIMUM FLEXIBILITY

    Ideally I would like to move transpose_forward/backward into the configurations so that this can also be done
    differently for each configuration but this would cause problems with identifying the correct axes for 2d. There
    surely is a way around that but eh. I'm feeling lazy and featuritis must also not be pushed to the extremes.

    So for now if you want a different transpose_forward/backward you need to create a new planner. Also not too
    hard.
    """
    # we use this as a cache to prevent having to instantiate the architecture too often. Saves computation time
    _tmp = {}

    # first get transpose
    transpose_forward, transpose_backward = self.determine_transpose()

    # get fullres spacing and transpose it
    fullres_spacing = self.determine_fullres_target_spacing()
    fullres_spacing_transposed = fullres_spacing[transpose_forward]

    # get transposed new median shape (what we would have after resampling)
    new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                    zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
    new_median_shape = np.median(new_shapes, 0)
    new_median_shape_transposed = new_median_shape[transpose_forward]

    approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                            self.dataset_json['numTraining'])
    # only run 3d if this is a 3d dataset
    if new_median_shape_transposed[0] != 1:
        plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                            new_median_shape_transposed,
                                                            self.generate_data_identifier('3d_fullres'),
                                                            approximate_n_voxels_dataset, _tmp)
        # maybe add 3d_lowres as well
        patch_size_fullres = plan_3d_fullres['patch_size']
        median_num_voxels = np.prod(new_median_shape_transposed, dtype=np.float64)
        num_voxels_in_patch = np.prod(patch_size_fullres, dtype=np.float64)

        plan_3d_lowres = None
        lowres_spacing = deepcopy(plan_3d_fullres['spacing'])

        spacing_increase_factor = 1.03  # used to be 1.01 but that is slow with new GPU memory estimation!
        while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:
            # we incrementally increase the target spacing. We start with the anisotropic axis/axes until it/they
            # is/are similar (factor 2) to the other ax(i/e)s.
            max_spacing = max(lowres_spacing)
            if np.any((max_spacing / lowres_spacing) > 2):
                lowres_spacing[(max_spacing / lowres_spacing) > 2] *= spacing_increase_factor
            else:
                lowres_spacing *= spacing_increase_factor
            median_num_voxels = np.prod(plan_3d_fullres['spacing'] / lowres_spacing * new_median_shape_transposed,
                                        dtype=np.float64)
            # print(lowres_spacing)
            plan_3d_lowres = self.get_plans_for_configuration(lowres_spacing,
                                                                tuple([round(i) for i in plan_3d_fullres['spacing'] /
                                                                        lowres_spacing * new_median_shape_transposed]),
                                                                self.generate_data_identifier('3d_lowres'),
                                                                float(np.prod(median_num_voxels) *
                                                                    self.dataset_json['numTraining']), _tmp)
            num_voxels_in_patch = np.prod(plan_3d_lowres['patch_size'], dtype=np.int64)
            print(f'Attempting to find 3d_lowres config. '
                    f'\nCurrent spacing: {lowres_spacing}. '
                    f'\nCurrent patch size: {plan_3d_lowres["patch_size"]}. '
                    f'\nCurrent median shape: {plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed}')
        if np.prod(new_median_shape_transposed, dtype=np.float64) / median_num_voxels < 2:
            print(f'Dropping 3d_lowres config because the image size difference to 3d_fullres is too small. '
                    f'3d_fullres: {new_median_shape_transposed}, '
                    f'3d_lowres: {[round(i) for i in plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed]}')
            plan_3d_lowres = None
        if plan_3d_lowres is not None:
            plan_3d_lowres['batch_dice'] = False
            plan_3d_fullres['batch_dice'] = True
        else:
            plan_3d_fullres['batch_dice'] = False
    else:
        plan_3d_fullres = None
        plan_3d_lowres = None

    # 2D configuration
    plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
                                                new_median_shape_transposed[1:],
                                                self.generate_data_identifier('2d'), approximate_n_voxels_dataset,
                                                _tmp)
    plan_2d['batch_dice'] = True

    print('2D U-Net configuration:')
    print(plan_2d)
    print()

    # median spacing and shape, just for reference when printing the plans
    median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
    median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

    # instead of writing all that into the plans we just copy the original file. More files, but less crowded
    # per file.
    shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

    # json is ###. I hate it... "Object of type int64 is not JSON serializable"
    plans = {
        'dataset_name': self.dataset_name,
        'plans_name': self.plans_identifier,
        'original_median_spacing_after_transp': [float(i) for i in median_spacing],
        'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
        'image_reader_writer': self.determine_reader_writer().__name__,
        'transpose_forward': [int(i) for i in transpose_forward],
        'transpose_backward': [int(i) for i in transpose_backward],
        'configurations': {'2d': plan_2d},
        'experiment_planner_used': self.__class__.__name__,
        'label_manager': 'LabelManager',
        'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
            'foreground_intensity_properties_per_channel']
    }

    if plan_3d_lowres is not None:
        plans['configurations']['3d_lowres'] = plan_3d_lowres
        if plan_3d_fullres is not None:
            plans['configurations']['3d_lowres']['next_stage'] = '3d_cascade_fullres'
        print('3D lowres U-Net configuration:')
        print(plan_3d_lowres)
        print()
    if plan_3d_fullres is not None:
        plans['configurations']['3d_fullres'] = plan_3d_fullres
        print('3D fullres U-Net configuration:')
        print(plan_3d_fullres)
        print()
        if plan_3d_lowres is not None:
            plans['configurations']['3d_cascade_fullres'] = {
                'inherits_from': '3d_fullres',
                'previous_stage': '3d_lowres'
            }

    self.plans = plans
    self.save_plans(plans)
    return plans
