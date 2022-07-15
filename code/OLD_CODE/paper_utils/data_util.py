import errno
import glob
import os

import cv2
from rich.console import Console
from tqdm import tqdm

console = Console()


class DataConverter:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        pass

    @staticmethod
    def _check_folder(folder_path: str):
        return os.path.exists(os.path.dirname(folder_path))

    @staticmethod
    def _check_create_folder(folder_path: str):
        if not os.path.exists(os.path.dirname(folder_path)):
            try:
                os.makedirs(os.path.dirname(folder_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise


class FrameToFramesRescaler(DataConverter):
    def __init__(self, input_paths: str, output_path: str, final_resolution_x: int = None,
                 final_resolution_y: int = None):
        super().__init__(input_paths, output_path)
        """
        Init of the class
        :param input_paths: paths of the folders containing the frames
        :param output_path: path of the output folder
        :param final_resolution_x: optional, use it if you want to resize the frames
        :param final_resolution_y: optional, use it if you want to resize the frames
        """
        # assert type(final_resolution_x) is int, "final_resolution_x is not an integer: %r" % final_resolution_x
        # assert type(final_resolution_y) is int, "final_resolution_y  is not an integer: %r" % final_resolution_y
        assert self._check_folder(input_paths), "input paths don't exist"
        self._input_paths = glob.glob(input_paths + "/*")
        self._output_path = output_path
        self._final_resolution = (final_resolution_x, final_resolution_y)

    def rescale(self):
        """
        Call for rescaling frames
        :return:
        """
        for i, input_path in enumerate(self._input_paths):
            for frame_file in tqdm(glob.glob(input_path + "/*"), desc=f"Rescaling Frames {i}/{len(self._input_paths)}"):
                set_name = frame_file.split("/")[-2]
                data_path = os.path.join(self._output_path, set_name + "/")
                self._check_create_folder(data_path)
                self._rescaler_helper(frame_file, data_path)
        print("Completed")

    def current_settings(self):
        """
        print current settings
        :return: nothing
        """
        print(f"{self._input_paths=}\n{self._output_path=}\n{self._final_resolution=}")

    def change_resolution(self, new_resolution_x: int, new_resolution_y: int):
        """
        change resolution after class init for resizing final frames
        :param new_resolution_x: final width of the frames
        :param new_resolution_y: final height of the frames

        """
        assert type(new_resolution_x) is int, "new_resolution_x is not an integer: %r" % new_resolution_x
        assert type(new_resolution_y) is int, "new_resolution_y is not an integer: %r" % new_resolution_y
        self._final_resolution = (new_resolution_x, new_resolution_y)

    def change_input_paths(self, new_paths: str):
        """
        change input path after class init
        :param new_paths: new input path
        """
        assert self._check_folder(new_paths), "new input path doesn't exists"
        self._input_paths = new_paths

    def change_output_path(self, new_path: str):
        """
        change output path after class init
        :param new_path: new output path
        """
        self._output_path = new_path

    def _rescaler_helper(self, file: str, subfolder: str):
        image = cv2.imread(file)
        count = file.split("/")[-1]
        if '_' in count:
            count = count.split("_")[0]

        height, width, _ = image.shape
        if self._final_resolution[0] is None or self._final_resolution[1] is None:
            self._final_resolution = (width, height)

        if width > self._final_resolution[0]:
            frame = cv2.resize(image, self._final_resolution, fx=0, fy=0, interpolation=cv2.INTER_AREA)
        else:
            frame = image

        cv2.imwrite(
            os.path.join(
                subfolder,
                f"{count}".zfill(6) +
                f"_{self._final_resolution[0]}_{self._final_resolution[1]}.jpg"),
            frame)  # save frame as JPEG file

