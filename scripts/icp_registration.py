import numpy as np
import open3d as o3d
import os
import copy

from realsense_api import RealsenseAPI
from scipy.spatial.transform import Rotation as R


# check opencv python package
with_opencv = True


def pairwise_registration(
    source,
    target,
    init_source_pose=np.identity(4),
    init_target_pose=np.identity(4),
    max_correspondence_distance_coarse=0.1,
    max_correspondence_distance_fine=0.02,
):
    print("Apply point-to-plane ICP")
    init_pose = init_source_pose @ np.linalg.inv(init_target_pose)
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    target_copy.transform(init_pose)
    print(f"Initial transformation:")
    o3d.visualization.draw_geometries([source_copy, target_copy])

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_coarse,
        init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    transformation_icp = icp_fine.transformation
    print(f"After ICP")
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    target_temp.transform(transformation_icp)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    information_icp = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine, icp_fine.transformation
        )
    )
    return transformation_icp, information_icp


def full_registration(
    pcds,
    init_poses,
):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print(f"Pairwise registration: {source_id} -> {target_id}")
            print(f"Source pose: {init_poses[source_id]}")
            print(f"Target pose: {init_poses[target_id]}")

            source_pose = init_poses[source_id]
            target_pose = init_poses[target_id]
            source_T_target = np.linalg.inv(source_pose) @ target_pose

            [success_hybrid_term, trans_hybrid_term, info] = (
                o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd_images[0],
                    rgbd_images[1],
                    intrinsics_[0],
                    np.linalg.inv(source_T_target),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option,
                )
            )
            # transformation_icp, information_icp = refine(
            #     pcds[source_id], pcds[target_id], source_T_target
            # )
            # transformation_icp, information_icp = pairwise_registration(
            #     pcds[source_id],
            #     pcds[target_id],
            #     init_poses[source_id],
            #     init_poses[target_id],
            #     max_correspondence_distance_coarse=max_correspondence_distance_coarse,
            #     max_correspondence_distance_fine=max_correspondence_distance_fine,
            # )
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False,
                    )
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    return pose_graph


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )
    return result


def execute_fast_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 0.5
    print(
        ":: Apply fast global registration with distance threshold %.3f"
        % distance_threshold
    )
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def refine_registration(source, target, voxel_size, transformation_init):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        transformation_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def refine(source, target, init_transformation):
    result_icp_prev = refine_registration(
        source,
        target,
        0.025,
        np.linalg.inv(init_transformation),
    )

    iter = 30
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source,
        target,
        0.01,
        result_icp_prev.transformation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
        ),
    )

    information_icp = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, 0.025 * 4, result_icp.transformation
        )
    )
    return result_icp.transformation, information_icp


def register_one_rgbd_pair(
    s,
    t,
    source_rgbd_image,
    target_rgbd_image,
    intrinsic,
    with_opencv,
    config,
    init_pose=None,
):

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = config["depth_diff_max"]
    odo_init = init_pose if init_pose is not None else np.identity(4)
    [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image,
        target_rgbd_image,
        intrinsic,
        odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )
    return [success, trans, info]


def make_posegraph_for_fragment(
    rgbd_frames, intrinsic, with_opencv, config, init_poses=None
):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    num_frames = len(rgbd_frames)
    for s in range(0, num_frames):
        for t in range(s + 1, num_frames):
            # odometry
            source_rgbd_image = rgbd_frames[s]
            target_rgbd_image = rgbd_frames[t]
            if init_poses is not None:
                init_pose = np.linalg.inv(init_poses[s]) @ init_poses[t]
            if t == s + 1:
                [success, trans, info] = register_one_rgbd_pair(
                    s,
                    t,
                    source_rgbd_image,
                    target_rgbd_image,
                    intrinsic,
                    with_opencv,
                    config,
                    init_pose=np.linalg.inv(init_pose),
                )
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(trans_odometry_inv)
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        s, t, trans, info, uncertain=False
                    )
                )

            # keyframe loop closure
            else:
                [success, trans, info] = register_one_rgbd_pair(
                    s,
                    t,
                    source_rgbd_image,
                    target_rgbd_image,
                    intrinsic,
                    with_opencv,
                    config,
                )
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s, t, trans, info, uncertain=True
                        )
                    )

    return pose_graph


def run_posegraph_optimization(
    pose_graph, max_correspondence_distance, preference_loop_closure
):
    # to display messages from o3d.pipelines.registration.global_optimization
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=preference_loop_closure,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    return pose_graph


def integrate_rgb_frames_for_fragment(rgbd_frames, pose_graph, intrinsic, config):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in range(len(pose_graph.nodes)):
        rgbd = rgbd_frames[i]
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(rgbd_frames, pose_graph, intrinsic, config):
    mesh = integrate_rgb_frames_for_fragment(
        rgbd_frames,
        pose_graph,
        intrinsic,
        config,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    return pcd


if __name__ == "__main__":
    cams = RealsenseAPI(height=480, width=848, fps=30)

    print(f"Num cameras: {cams.get_num_cameras()}")

    output_dir = "./outputs_extrinsics/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "camera1"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "camera2"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "camera3"), exist_ok=True)

    rgbd = cams.get_rgbd()

    poses = {
        "cam0": {
            "R": [
                [0.9988507394834056, 0.03028266805074297, -0.0371497265828584],
                [-0.016023030054223016, -0.5195245701970147, -0.8543052385853411],
                [-0.04517083845727273, 0.8539187003880233, -0.518442319799712],
            ],
            "t": [-0.4982499301603227, 0.2319365438244401, 1.000725660647518],
            # "R": [
            #     [0.9992051124572754, 0.024900231510400772, -0.031129786744713783],
            #     [-0.013922946527600288, -0.5137407183647156, -0.8578324913978577],
            #     [-0.03735286742448807, 0.8575840592384338, -0.5129857063293457],
            # ],
            # "t": [-0.5161314600444653, 0.24456735918345468, 0.9599211043543104],
        },
        "cam1": {
            "R": [
                [-0.08352549597888383, 0.9621374601875166, 0.25945125675918435],
                [0.8405989904429897, 0.20785610682788822, -0.500189097104278],
                [-0.5351791949368981, 0.17631594497748634, -0.8261330972203766],
            ],
            "t": [-0.10744608395131783, -0.28546667284889027, 1.1970658672018013],
            # "R": [
            #     [-0.08679603343254853, 0.9652158032731323, 0.2466269602417027],
            #     [0.8188622639618721, 0.21011827837246622, -0.5341487248595296],
            #     [-0.5673896219928694, 0.1555915425099572, -0.8086162650056707],
            # ],
            # "t": [-0.07427998068744979, -0.25430493610422233, 1.2170195536490946],
        },
        "cam2": {
            "R": [
                [-0.9805212736543321, -0.1963001601110038, 0.00664994297289015],
                [-0.15135279142159733, 0.7335638345891166, -0.6625529624086519],
                [0.12518109420265916, -0.6506537586292271, -0.7489855352947886],
            ],
            "t": [
                0.5882456482370338,
                0.10036606237908585,
                0.8854745051400028,
            ],  # "R": [
            #     [-0.9870733973616255, -0.1601499795264944, 0.006167226222480767],
            #     [-0.1254097045167346, 0.7478494777111884, -0.6519153009037237],
            #     [0.09979206451217507, -0.6442616771466401, -0.7582666995109131],
            # ],
            # "t": [0.5713117438376824, 0.08066768097835486, 0.8995083374565274],
        },
    }
    rgbd_images = []
    intrinsics_ = []
    for i, device_id in enumerate(cams.device_ls):
        color_image = o3d.geometry.Image(
            np.ascontiguousarray(rgbd[i, :, :, :3]).astype(np.uint8)
        )
        depth_image = o3d.geometry.Image(rgbd[i, :, :, 3].astype(np.uint16))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            convert_rgb_to_intensity=False,
        )
        rgbd_images.append(rgbd_image)

        intrinsics = cams.get_intrinsics()[i]
        fx = intrinsics.fx
        fy = intrinsics.fy
        ppx = intrinsics.ppx
        ppy = intrinsics.ppy
        intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=cams.width, height=cams.height, fx=fx, fy=fy, cx=ppx, cy=ppy
        )
        intrinsics_.append(intrinsics_o3d)

    cam0_T_w = np.eye(4)
    cam0_T_w[:3, :3] = np.array(poses["cam0"]["R"])
    cam0_T_w[:3, 3] = np.array(poses["cam0"]["t"])

    cam1_T_w = np.eye(4)
    cam1_T_w[:3, :3] = np.array(poses["cam1"]["R"])
    cam1_T_w[:3, 3] = np.array(poses["cam1"]["t"])

    cam2_T_w = np.eye(4)
    cam2_T_w[:3, :3] = np.array(poses["cam2"]["R"])
    cam2_T_w[:3, 3] = np.array(poses["cam2"]["t"])

    w_T_cam0 = np.linalg.inv(cam0_T_w)
    w_T_cam1 = np.linalg.inv(cam1_T_w)
    w_T_cam2 = np.linalg.inv(cam2_T_w)

    cam0_T_cam1 = cam0_T_w @ w_T_cam1
    cam0_T_cam2 = cam0_T_w @ w_T_cam2
    cam1_T_cam2 = cam1_T_w @ w_T_cam2

    config = {
        "depth_diff_max": 0.07,
        "voxel_size": 0.02,
        "depth_max": 1.5,
        "preference_loop_closure_odometry": 0.1,
        "tsdf_cubic_size": 2.0,
    }
    pose_graph = make_posegraph_for_fragment(
        rgbd_images,
        intrinsics_[0],
        with_opencv,
        config,
        init_poses=[w_T_cam0, w_T_cam1, w_T_cam2],
    )
    run_posegraph_optimization(
        pose_graph, config["depth_diff_max"], config["preference_loop_closure_odometry"]
    )
    pcd = make_pointcloud_for_fragment(rgbd_images, pose_graph, intrinsics_[0], config)
    o3d.visualization.draw_geometries([pcd])

    print(f"cam0_T_cam1: {cam0_T_cam1}")
    print(f"cam0_T_cam2: {cam0_T_cam2}")

    print(f"After optimization cam0_T_cam1: {pose_graph.nodes[1].pose}")
    print(f"After optimization cam0_T_cam2: {pose_graph.nodes[2].pose}")

    cam0_T_cam1_optimized = pose_graph.nodes[1].pose
    cam0_T_cam2_optimized = pose_graph.nodes[2].pose

    w_T_cam1_optimized = w_T_cam0 @ cam0_T_cam1_optimized
    w_T_cam2_optimized = w_T_cam0 @ cam0_T_cam2_optimized

    print(f"w_T_cam1_optimized: {w_T_cam1_optimized}")
    print(f"w_T_cam2_optimized: {w_T_cam2_optimized}")

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
    camera_tfs = [left_camera_tf, right_camera_tf, front_camera_tf]

    w_T_cam0_optimized = w_T_cam0.copy() @ left_camera_tf
    w_T_cam1_optimized = w_T_cam1_optimized.copy() @ right_camera_tf
    w_T_cam2_optimized = w_T_cam2_optimized.copy() @ front_camera_tf

    w_q_cam0_optimized = R.from_matrix(w_T_cam0_optimized[:3, :3]).as_quat(
        canonical=True
    )
    w_q_cam1_optimized = R.from_matrix(w_T_cam1_optimized[:3, :3]).as_quat(
        canonical=True
    )
    w_q_cam2_optimized = R.from_matrix(w_T_cam2_optimized[:3, :3]).as_quat(
        canonical=True
    )

    w_t_cam0_optimized = w_T_cam0_optimized[:3, 3]
    w_t_cam1_optimized = w_T_cam1_optimized[:3, 3]
    w_t_cam2_optimized = w_T_cam2_optimized[:3, 3]

    print(f"w_T_cam0_optimized: {w_t_cam0_optimized}, {w_q_cam0_optimized}")
    print(f"w_T_cam1_optimized: {w_t_cam1_optimized}, {w_q_cam1_optimized}")
    print(f"w_T_cam2_optimized: {w_t_cam2_optimized}, {w_q_cam2_optimized}")
