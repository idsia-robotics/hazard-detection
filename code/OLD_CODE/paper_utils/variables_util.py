# unifromed model variables
dataset_names = {0: "tunnel", 1: "factory", 2: "corridor"}
available_scale_levels = {0: "s1", 1: "s2", 2: "s4", 3: "s8"}
scaled_image_shapes = {0: (512, 512), 1: (256, 256), 2: (128, 128), 3: (64, 64)}
datasets_labels_names = {
    0:
        {
            0: "normal",
            1: "dust",
            2: "root",
            3: "wet",
        },
    1:
        {
            0: "normal",
            1: "mist",
            2: "tape",
        },
    2:
        {
            0: "normal",
            1: "water",
            2: "cellophane",
            3: "cable",
            4: "defects",
            5: "hanging cable",
            6: "floor",
            7: "human",
            8: "screws",
        },
}
