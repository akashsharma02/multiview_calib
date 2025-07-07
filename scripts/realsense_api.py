from collections import OrderedDict
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import argparse
import cv2


class RealsenseAPI:
    """Wrapper that implements boilerplate code for RealSense cameras"""

    def __init__(self, height=480, width=640, fps=30, warm_start=60):
        self.height = height
        self.width = width
        self.fps = fps

        # Identify devices
        self.device_ls = []
        for c in rs.context().query_devices():
            self.device_ls.append(c.get_info(rs.camera_info(1)))

        # Start stream
        print(f"Connecting to RealSense cameras ({len(self.device_ls)} found) ...")
        self.pipes = []
        self.profiles = OrderedDict()
        for i, device_id in enumerate(self.device_ls):
            pipe = rs.pipeline()
            config = rs.config()

            config.enable_device(device_id)
            config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )

            self.pipes.append(pipe)
            self.profiles[device_id] = pipe.start(config)

            print(f"Connected to camera {i+1} ({device_id}).")

        self.align = rs.align(rs.stream.color)
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, 0.1)  # 10 cm
        self.threshold_filter.set_option(rs.option.max_distance, 1.5)  # 1 m

        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        for _ in range(warm_start):
            self._get_frames()
        self.setup_tags()

    def _get_frames(self):
        framesets = [pipe.wait_for_frames() for pipe in self.pipes]
        return [self.align.process(frameset) for frameset in framesets]

    def get_intrinsics(self):
        intrinsics_ls = []
        for profile in self.profiles.values():
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()

            intrinsics_ls.append(intrinsics)

        return intrinsics_ls

    def setup_tags(self):
        """
        Hand-measure the transformations to the allegro base frame (at the tip of the middle finger)
        """
        self.transform_dict = {
            1: np.eye(4),
        }
        self.transform_dict[1][:3, :3] = R.from_quat(
            [0.7071068, 0, 0.7071068, 0]
        ).as_matrix()  # (X: 0, Y: 90, Z: 180) in YZX euler
        self.transform_dict[1][:3, 3] = np.array(
            [0.0237, -0.031, -0.0112]
        )  # hand measured from aruco to allegro base frame

    def marker_to_cam_pose(self, marker):
        """
        Returns camera pose in allegro frame
        """
        # Transform to the allegro base frame
        allegro_pose = marker.pose.matrix() @ self.transform_dict[marker.id]
        return np.linalg.inv(allegro_pose)

    def get_intrinsics_dict(self):
        intrinsics_ls = OrderedDict()
        for device_id, profile in self.profiles.items():
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()
            param_dict = dict(
                [
                    (p, getattr(intrinsics, p))
                    for p in dir(intrinsics)
                    if not p.startswith("__")
                ]
            )
            param_dict["model"] = param_dict["model"].name

            intrinsics_ls[device_id] = param_dict

        return intrinsics_ls

    def get_num_cameras(self):
        return len(self.device_ls)

    def get_rgbd(self):
        """Returns a numpy array of [n_cams, height, width, RGBD]"""
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()

        rgbd = np.empty([num_cams, self.height, self.width, 4], dtype=np.uint16)

        # return framesets[0].get_color_frame(), framesets[0].get_depth_frame()
        for i, frameset in enumerate(framesets):
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            filtered_depth_frame = self.threshold_filter.process(depth_frame)
            rgbd[i, :, :, :3] = np.asanyarray(color_frame.get_data())

            rgbd[i, :, :, 3] = np.asanyarray(filtered_depth_frame.get_data())

        return rgbd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture RGBD data from RealSense cameras."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Height of the camera stream."
    )
    parser.add_argument(
        "--width", type=int, default=848, help="Width of the camera stream."
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="output",
        help="Folder to save the RGBD data.",
    )

    args = parser.parse_args()

    cams = RealsenseAPI(height=args.height, width=args.width, fps=30)

    print(f"Num cameras: {cams.get_num_cameras()}")

    rgbd = cams.get_rgbd()

    for i, device_id in enumerate(cams.device_ls):

        color_image = np.ascontiguousarray(rgbd[i, :, :, :3]).astype(np.uint8)
        depth_image = np.ascontiguousarray(rgbd[i, :, :, 3]).astype(np.uint16)

        cv2.imwrite(args.output_folder + f"/camera{i}.png", color_image[..., ::-1])
        # cv2.imwrite(args.output_folder + f"/depth_{i}.png", depth_image)
