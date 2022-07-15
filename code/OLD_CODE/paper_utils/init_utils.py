def uniformed_model_paths_init():
    model_save_prefix = "path/to/experiment/folder"
    noise_path = f"path/to/perlin_noise"
    train_path = f"path/to/train_set"
    val_path = f"path/to/validation_set"
    test_paths = f"path/to/test_set"
    qualitative_paths = f"path/to/qualitative_set"
    test_labels_csv = f"path/to/frames_labels.csv"
    return model_save_prefix, qualitative_paths, test_paths, test_labels_csv, train_path, val_path, noise_path
