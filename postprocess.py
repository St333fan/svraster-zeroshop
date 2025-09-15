import pymeshlab
import os
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import json

def postprocess_mesh(mesh_path="", bundler_path="", txt_path="", euler_angles=None, texturize=True, estimated_height=None):

    # Find the folder that matches the pattern obj_XXXXXX in the path
    path_parts = os.path.normpath(mesh_path).split(os.sep)
    folder_name = None
    for part in reversed(path_parts):
        if part.startswith("obj_") and part[4:].isdigit():
            folder_name = part
            break
    if folder_name is None:
        # Fallback to immediate parent folder
        folder_name = os.path.basename(os.path.dirname(mesh_path))

    ms = pymeshlab.MeshSet()

    # Load project rasters/images
    ms.load_project([bundler_path, txt_path])
    print(f"Loaded rasters: {ms.raster_number()}")

    # Load mesh
    ms.load_new_mesh(mesh_path)
    print(f"Number of meshes: {ms.mesh_number()}")
    print(f"Vertices in current mesh: {ms.current_mesh().vertex_number()}")

    # Remove small disconnected components (floaters)
    print("Removing floaters...")
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=5000)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=50000, preservetopology=True)
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=5000)

    # Repair non-manifold geometry
    print("Repairing non-manifold edges and vertices...")
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    # Smooth the mesh
    print("Smoothing the mesh...")
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=1)

    # Close small holes
    print("Closing holes...")
    ms.meshing_close_holes(maxholesize=30)

    # Repair non-manifold geometry again after hole closing
    print("Repairing non-manifold edges and vertices...")
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    
    if texturize:
        # Set vertex color to white
        print("Setting vertex color to white...")
        ms.set_color_per_vertex(color1=pymeshlab.Color(255, 255, 255, 255))
        print(f"Vertices after coloring: {ms.current_mesh().vertex_number()}")
        
        # Generate texture from images ! pip install pymeshlab==2023.12.post3
        print("Generating texture from images...")
        ms.compute_texcoord_parametrization_and_texture_from_registered_rasters(
            texturename=f"{folder_name}.png",
            texturesize=2048,
        )

    # Center the Mesh
    print("Centering the mesh at the origin...")
    ms.compute_matrix_from_translation(
        traslmethod='Center on Layer BBox'
    )

    # Euler angle data ---
    #euler_angles = [-110.8115705, 36.52771323, 167.92819583]

    if euler_angles is not None:
        # Apply the rotation directly using the filter
        print("Applying rotation directly from Euler angles...")
        ms.compute_matrix_from_translation_rotation_scale(
            rotationx = euler_angles[0],
            rotationy = euler_angles[1],
            rotationz = euler_angles[2]
        )

    # Scale mesh to match estimated_height if provided
    if estimated_height is not None:
        # Object Detection and Pose Estimation uses mm as unit
        estimated_height = float(estimated_height) * 1000.0  # Convert meters to millimeters
    else:
        estimated_height = 200.0  # Default height in mm if not provided

    print("Scaling mesh to match estimated height in mm; 1m = 1000mm ...")
    bbox = ms.current_mesh().bounding_box()
    current_height = bbox.max()[2] - bbox.min()[2]

    if current_height > 0:
        scale_factor = estimated_height / current_height
        print(f"Current height: {current_height}, scale factor: {scale_factor}")
        ms.compute_matrix_from_scaling_or_normalization(
            axisx=scale_factor,
            uniformflag=True,
            scalecenter='origin',
            freeze=True
        )
    else:
        print("Warning: Current mesh height is zero, skipping scaling.")

    # Save mesh and texture using the parent folder name (e.g., obj_000001)
    save_dir = os.path.dirname(mesh_path)
    mesh_save_path = os.path.join(save_dir, f"{folder_name}.ply")
    if texturize:
        ms.save_current_mesh(mesh_save_path, save_textures=True)
    else:
        ms.save_current_mesh(mesh_save_path, save_textures=False)

def load_camera_info(filename):
    """
    Loads camera information from the specified file.
    File format:
    - First line: (ignored and header)
    - Second line: number of cameras (int)
    - For each camera:
        - 2 floats: f, two distortion values
        - 3x3 rotation matrix (row-major, 9 floats)
        - 1x3 translation vector (3 floats)
    Returns:
        cameras: list of dicts with keys 'f', 'dist', 'R', 't'
    """
    cameras = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip first line (header or comment)
    idx = 1
    num_cameras = int(lines[idx].strip().split()[0])
    idx += 1
    for _ in range(num_cameras):
        # f and two distortion values
        f_val, d1, d2 = map(float, lines[idx].strip().split())
        idx += 1
        # 3x3 rotation matrix
        R = []
        for _ in range(3):
            R.append(list(map(float, lines[idx].strip().split())))
            idx += 1
        R = np.array(R)
        # 1x3 translation vector
        t = np.array(list(map(float, lines[idx].strip().split())))
        idx += 1
        # Compute camera center: C = -R.T @ t
        C = -R.T @ t
        cameras.append({
            'f': f_val,
            'dist': [d1, d2],
            'R': R,
            't': t,
            'center': C
        })
    return cameras, lines, idx

def calculate_rotation(bundler_path="", txt_path=""):
    cameras, lines, idx = load_camera_info(bundler_path)

    with open(txt_path) as f:
        lines_txt = [line.strip() for line in f if line.strip()]

    # Extract numeric part and find the index of the minimum
    numbers = [int(name.split('.')[0]) for name in lines_txt]
    cam0_idx = numbers.index(min(numbers))
    print(f"Index of the lowest number: {cam0_idx}")


    # Load 3D points
    points = []
    colors = []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        try:
            float(line.split()[0])
        except Exception:
            break
        xyz = list(map(float, line.split()))
        idx += 1
        rgb = list(map(int, lines[idx].strip().split()))
        idx += 1
        idx += 1  # skip visibility
        points.append(xyz)
        colors.append(rgb)
        if len(points) > 1000:
            break

    points = np.array(points)
    colors = np.array(colors) / 255.0

    # Camera centers
    centers = np.array([cam['center'] for cam in cameras])

    # Average viewing direction 
    directions = np.array([-cam['R'][2] for cam in cameras])
    avg_direction = np.mean(directions, axis=0)
    avg_direction /= np.linalg.norm(avg_direction)
    centroid = np.mean(centers, axis=0)

    # Camera 0 viewing direction
    cam0_view_dir = -cameras[cam0_idx]['R'][2]
    cam0_view_dir = cam0_view_dir / np.linalg.norm(cam0_view_dir)

    print(f"Average direction: {avg_direction}")
    print(f"Cam0 direction: {cam0_view_dir}")

    # Step 1: Create perpendicular vector using right-hand rule
    # Cross product: cam0_view_dir × avg_direction gives perpendicular vector
    perp_vec1 = np.cross(cam0_view_dir, avg_direction)
    if np.linalg.norm(perp_vec1) < 1e-10:
        # Vectors are parallel, choose arbitrary perpendicular
        perp_vec1 = np.array([1, 0, 0]) if abs(avg_direction[0]) < 0.9 else np.array([0, 1, 0])
        perp_vec1 = perp_vec1 - np.dot(perp_vec1, avg_direction) * avg_direction
    perp_vec1 = perp_vec1 / np.linalg.norm(perp_vec1)

    # Step 2: Create second perpendicular vector 
    # Cross product: perp_vec1 × avg_direction gives second perpendicular vector
    perp_vec2 = np.cross(perp_vec1, avg_direction)
    perp_vec2 = np.cross(avg_direction, perp_vec1)
    perp_vec2 = perp_vec2 / np.linalg.norm(perp_vec2)

    print(f"Perpendicular vec1 (X): {perp_vec1}")
    print(f"Perpendicular vec2 (Y): {perp_vec2}")
    print(f"Average direction (Z): {avg_direction}")

    # Step 3: Create initial coordinate system
    # X = perp_vec1, Y = perp_vec2, Z = avg_direction
    initial_x = perp_vec1.copy()
    initial_y = perp_vec2.copy() 
    initial_z = avg_direction.copy()

    # Step 4: Apply 180° rotation around Y-axis (green vector) to flip Y and Z
    # After rotation: X stays same, Y becomes -Y, Z becomes -Z
    final_x = -initial_y.copy()
    final_y = -initial_x.copy()
    final_z = -initial_z.copy()

    # Final rotation matrix (world to new coordinate system)
    R = np.column_stack([final_x, final_y, final_z])
    
    # Check determinant
    det = np.linalg.det(R)
    print(f"Determinant of rotation matrix: {det}")
    if det < 0:
        R[:, 0] *= -1  # Flip the sign of the first column
        print("Corrected rotation matrix to be right-handed.")
        det = np.linalg.det(R)
        print(f"New determinant: {det}")
    if det > 0:
        r = Rotation.from_matrix(R)
        angles_deg = r.as_euler('xyz', degrees=True)
        print("Euler angles (degrees, XYZ order):", angles_deg)
    else:
        print("Invalid rotation matrix: determinant is non-positive even after correction.")
        angles_deg = np.array([0, 0, 0])
    return angles_deg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess mesh with optional rotation from Bundler output.")
    parser.add_argument("--mesh", type=str, required=False, help="Path to input mesh file (e.g., fuse_post.ply)")
    parser.add_argument("--bundler", type=str, required=False, help="Path to Bundler output file (e.g., images/scene.bundler.out)")
    parser.add_argument("--bundler_txt", type=str, required=False, help="Path to image list txt file (e.g., images/scene.list.txt)")
    parser.add_argument("--no-rotation", action="store_true", help="Skip rotation calculation and application")
    parser.add_argument("--object_info_json", type=str, required=False, help="Path to object_info.json file with estimated_height")
    parser.add_argument("--texture", action="store_true", help="Skip texturization step")
    args = parser.parse_args()

    # args.mesh = "/home/stefan/Projects/Grounded-SAM-2-zeroshop/dataset/obj_000001/train_pbr/mast3r-sfm/surface/2DGS_output/train/ours_30000/fuse_post.ply"
    # args.bundler = "/home/stefan/Projects/Grounded-SAM-2-zeroshop/dataset/obj_000001/train_pbr/mast3r-sfm/surface/images/scene.bundle.out"
    # args.bundler_txt = "/home/stefan/Projects/Grounded-SAM-2-zeroshop/dataset/obj_000001/train_pbr/mast3r-sfm/surface/images/scene.list.txt"
    # args.object_info_json = "/home/stefan/Projects/Grounded-SAM-2-zeroshop/dataset/obj_000001/scene/output/object_info.json"

    # Load JSON info if provided
    estimated_height = None
    if args.object_info_json:
        with open(args.object_info_json, 'r') as jf:
            obj_info = json.load(jf)
            estimated_height = obj_info.get("estimated_height", None)
        print(f"Loaded estimated_height from JSON: {estimated_height}")

    if args.no_rotation:
        rotation = None
    else:
        rotation = calculate_rotation(args.bundler, args.bundler_txt)

    if estimated_height is not None:
        postprocess_mesh(args.mesh, args.bundler, args.bundler_txt, euler_angles=rotation, estimated_height=estimated_height, texturize=args.texture)
    else:
        postprocess_mesh(args.mesh, args.bundler, args.bundler_txt, euler_angles=rotation, texturize=args.texture)