import argparse
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as Rot

from multiview_calib import utils

pio.renderers.default = "browser"


class CameraWireframe:

    def __init__(self, scale=1):
        self.scale = scale
        self.vertices = (
            np.array(
                [
                    [-1, -1, 0],  # Bottom center
                    [1, -1, 0],  # Top vertex 1
                    [1, 1, 0],  # Bottom vertex 2
                    [-1, 1, 0],  # Bottom vertex 3
                    [-1, -1, 3],  # Bottom center
                    [1, -1, 3],  # Top vertex 1
                    [1, 1, 3],  # Bottom vertex 2
                    [-1, 1, 3],  # Bottom vertex 3
                    [-5, -3, 5],  # Bottom center
                    [5, -3, 5],  # Top vertex 1
                    [5, 3, 5],  # Bottom vertex 2
                    [-5, 3, 5],  # Bottom vertex 3
                ]
            )
            * scale
        )

        self.edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 3),  # Bottom edges
            (0 + 4, 1 + 4),
            (1 + 4, 2 + 4),
            (2 + 4, 3 + 4),
            (0 + 4, 3 + 4),  # Top edges
            (0, 4),
            (1, 1 + 4),
            (2, 2 + 4),
            (3, 3 + 4),  # Bottom to top
            (0 + 8, 1 + 8),
            (1 + 8, 2 + 8),
            (2 + 8, 3 + 8),
            (0 + 8, 3 + 8),  # Top edges
            (4, 8),
            (5, 1 + 8),
            (6, 2 + 8),
            (7, 3 + 8),  # Top edges
        ]

        self.camera_position = np.array([0, 0, 0])
        self.camera_orientation = np.eye(3)

        self.vertices_system = (
            np.array([[0, 0, 0], [7.5, 0, 0], [0, 7.5, 0], [0, 0, 7.5]]) * self.scale
        )

        self.edges_system = [(0, 1), (0, 2), (0, 3)]

    def transform(self, R, t):
        self.camera_position = np.array(t).ravel()
        self.camera_orientation = np.array(R)
        self.vertices = (
            self.vertices @ self.camera_orientation.T
            + self.camera_position.reshape(1, 3)
        )
        self.vertices_system = (
            self.vertices_system @ self.camera_orientation.T
            + self.camera_position.reshape(1, 3)
        )
        return self

    def get_plotly_wireframe(self):
        edge_x = []
        edge_y = []
        edge_z = []
        x, y, z = zip(*self.vertices)
        for s, e in self.edges:
            # Add None between two edge points to draw separate lines
            edge_x += [x[s], x[e], None]
            edge_y += [y[s], y[e], None]
            edge_z += [z[s], z[e], None]

        return dict(x=edge_x, y=edge_y, z=edge_z)

    def get_plotly_coordinate_system(self):
        x, y, z = zip(*self.vertices_system)
        return (
            dict(
                x=[x[0], x[1], None], y=[y[0], y[1], None], z=[z[0], z[1], None]
            ),  # x-axis arrow
            dict(
                x=[x[0], x[2], None], y=[y[0], y[2], None], z=[z[0], z[2], None]
            ),  # y-axis arrow
            dict(x=[x[0], x[3], None], y=[y[0], y[3], None], z=[z[0], z[3], None]),
        )  # z-axis arrow


def main(filename_poses="poses.json", filename_points="points.json"):

    poses = utils.json_read(filename_poses)
    points = None
    if filename_points is not None:
        points = utils.json_read(filename_points)

    # estimate scale of the camera wireframes
    camera_positions = np.array(
        [np.dot(-np.array(x["R"]).T, np.array(x["t"])) for name, x in poses.items()]
    )
    scale = pdist(camera_positions).mean() / 50

    plotly_data = []

    # add camera wireframes to plot
    coordinate_systems = [dict(x=[], y=[], z=[]) for _ in range(3)]
    left_camera_tf = np.array(
        [
            [0.006, -1.000, -0.021, 0.015],
            [0.001, 0.021, -1.000, -0.000],
            [1.000, 0.006, 0.001, 0.000],
            [0.000, 0.000, 0.000, 1.000],
        ]
    )
    right_camera_tf = np.array(
        [
            [0.000, -1.000, 0.029, 0.015],
            [-0.003, -0.029, -1.000, 0.001],
            [1.000, -0.000, -0.003, 0.001],
            [0.000, 0.000, 0.000, 1.000],
        ]
    )
    front_camera_tf = np.array(
        [
            [-0.006, -1.000, 0.010, 0.015],
            [-0.005, -0.010, -1.000, 0.000],
            [1.000, -0.006, -0.005, 0.000],
            [0.000, 0.000, 0.000, 1.000],
        ]
    )
    camera_tfs = [right_camera_tf, front_camera_tf, left_camera_tf]
    for i, (camera_name, pose) in enumerate(poses.items()):
        R = np.array(pose["R"])
        t = np.array(pose["t"])

        R_inv = R.T
        t_inv = -R.T @ t.reshape(3, 1)

        camera_tf = camera_tfs[i]
        camera_R = camera_tf[:3, :3]
        camera_t = camera_tf[:3, 3]

        base_to_cam = np.eye(4)
        base_to_cam[:3, :3] = R_inv
        base_to_cam[:3, 3] = t_inv[:, 0]

        base_to_cam = base_to_cam @ camera_tf
        position = base_to_cam[:3, 3]
        rotation = base_to_cam[:3, :3]
        quaternion = Rot.from_matrix(rotation).as_quat(canonical=True)
        print(f"{camera_name}: translation: {position}")
        print(f"{camera_name}: quaternion: {quaternion}")
        camera = CameraWireframe(scale=scale).transform(R_inv, t_inv)

        plotly_data.append(
            go.Scatter3d(
                **camera.get_plotly_wireframe(),
                mode="lines",
                line=dict(width=4),
                name=camera_name,
            )
        )

        # combine arrows of the coordinate systems
        arrows = camera.get_plotly_coordinate_system()
        for i in range(3):
            coordinate_systems[i]["x"] += arrows[i]["x"]
            coordinate_systems[i]["y"] += arrows[i]["y"]
            coordinate_systems[i]["z"] += arrows[i]["z"]

    plotly_data.append(
        go.Scatter3d(
            **coordinate_systems[0],
            mode="lines",
            line=dict(color="red", width=6),
            name="camera systems x-axis",
        )
    )
    plotly_data.append(
        go.Scatter3d(
            **coordinate_systems[1],
            mode="lines",
            line=dict(color="green", width=6),
            name="camera systems y-axis",
        )
    )
    plotly_data.append(
        go.Scatter3d(
            **coordinate_systems[2],
            mode="lines",
            line=dict(color="blue", width=6),
            name="camera systems z-axis",
        )
    )

    # add points
    if points is not None:
        _points = np.array(points["points_3d"])
        plotly_data.append(
            go.Scatter3d(
                x=_points[:, 0],
                y=_points[:, 1],
                z=_points[:, 2],
                mode="markers",
                marker=dict(size=4, opacity=1),
                name="points",
            )
        )

    fig = go.Figure(data=plotly_data)
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data")
    )

    fig.show()


if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename_poses",
        "-p",
        type=str,
        required=True,
        default="global_poses.json",
        help="JSON file containing the camera poses.",
    )
    parser.add_argument(
        "--filename_points",
        "-o",
        type=str,
        required=False,
        default=None,
        help="JSON file containing the 3d points.",
    )

    args = parser.parse_args()
    main(**vars(args))

# python visualisation.py -p global_poses.json -o global_triang_points.json
