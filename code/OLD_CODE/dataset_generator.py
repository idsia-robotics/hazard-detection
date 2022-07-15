from paper_code_release.paper_utils.data_util import FrameToFramesRescaler

if __name__ == "__main__":
    s1_resolution = [512, 512]
    s2_resolution = [256, 256]
    s4_resolution = [128, 128]
    s8_resolution = [64, 64]
    # example
    vc_tunnel_s2 = FrameToFramesRescaler(
        input_paths="path/to/tunnel/s1_frames/",
        output_path="path/to/tunnel/s2_frames/",
        final_resolution_x=s2_resolution[0], final_resolution_y=s2_resolution[1])
    vc_tunnel_s2.rescale()

    vc_tunnel_s4 = FrameToFramesRescaler(
        input_paths="path/to/tunnel/s1_frames/",
        output_path="path/to/tunnel/s4_frames/",
        final_resolution_x=s4_resolution[0], final_resolution_y=s4_resolution[1])
    vc_tunnel_s4.rescale()

    vc_tunnel_s8 = FrameToFramesRescaler(
        input_paths="path/to/tunnel/s1_frames/",
        output_path="path/to/tunnel/s8_frames/",
        final_resolution_x=s8_resolution[0], final_resolution_y=s8_resolution[1])
    vc_tunnel_s8.rescale()
