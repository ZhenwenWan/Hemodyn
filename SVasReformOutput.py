import os
import re
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk

def parse_input_file(file_path):
    model_name = None
    segment_names = []

    with open(file_path, "r") as file:
        for line in file:
            stripped = line.strip()
            if stripped.upper().startswith("MODEL"):
                model_name = stripped.split()[1]
            elif stripped.upper().startswith("SEGMENT"):
                segment_names.append(stripped.split()[1])

    return model_name, segment_names

def parse_network_geometry(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    node_pattern = r"NODE\s+(\d+)\s+([-\.\dEe+]+)\s+([-\.\dEe+]+)\s+([-\.\dEe+]+)"
    nodes = pd.DataFrame(
        pd.DataFrame(
            re.findall(node_pattern, content),
            columns=["id", "x", "y", "z"]
        ).astype({"id": int, "x": float, "y": float, "z": float})
    ).set_index("id")

    segment_pattern = r"SEGMENT\s+(\S+)\s+(\d+)\s+[-\.\dEe+]+\s+\d+\s+(\d+)\s+(\d+)"
    segments = pd.DataFrame(
        re.findall(segment_pattern, content),
        columns=["name", "segment_id", "start_node", "end_node"]
    ).astype({"segment_id": int, "start_node": int, "end_node": int})

    return nodes, segments

def extract_segment_timeseries(file_path):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data[0], data[-1]  # first and last rows

def build_vtk_polydata(input_file_path):
    model, segment_names = parse_input_file(input_file_path)
    directory = os.path.dirname(input_file_path)
    nodes_df, segments_df = parse_network_geometry(input_file_path)

    points = vtk.vtkPoints()
    node_id_to_vtk_id = {}
    for node_id, row in nodes_df.iterrows():
        pid = points.InsertNextPoint(row["x"], row["y"], row["z"])
        node_id_to_vtk_id[node_id] = pid

    lines = vtk.vtkCellArray()
    segment_point_order = []
    for _, row in segments_df.iterrows():
        start_id = node_id_to_vtk_id[row["start_node"]]
        end_id = node_id_to_vtk_id[row["end_node"]]
        segment_point_order.append((start_id, end_id))
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, start_id)
        line.GetPointIds().SetId(1, end_id)
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    variable_names = ['area', 'flow', 'pressure', 'wss', 'Re']
    num_points = points.GetNumberOfPoints()
    num_timesteps = 80

    for var in variable_names:
        time_slices = [np.full(num_points, np.nan) for _ in range(num_timesteps)]
        for seg_idx, row in segments_df.iterrows():
            seg = row["name"]
            file_path = os.path.join(directory, f"{model}{seg}_{var}.dat")
            if not os.path.isfile(file_path):
                print(f"Missing: {file_path}")
                continue
            start_ts, end_ts = extract_segment_timeseries(file_path)
            start_pid, end_pid = segment_point_order[seg_idx]
            print(f"✔ {file_path} → start_point: {start_pid}, end_point: {end_pid}")
            for t in range(num_timesteps):
                time_slices[t][start_pid] = start_ts[t]
                time_slices[t][end_pid] = end_ts[t]

        for t in range(num_timesteps):
            vtk_arr = numpy_to_vtk(time_slices[t], deep=True)
            vtk_arr.SetName(f"{var}_t{t}")
            polydata.GetPointData().AddArray(vtk_arr)

    return polydata

def save_polydata_to_vtp(polydata, output_file):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()
    print(f"✅ Saved: {output_file}")

def convert_to_vtp(input_file, output_file):
    if not input_file.endswith(".in"):
        print("Reforming SimVascular OneDSolver results('model.in', 'output.vtp')")
        print("❌ Invalid input file. Expected '.in' extension.")
        print("Usage: convert_to_vtp('model.in', 'output.vtp')")
        return
    if not output_file.endswith(".vtp"):
        print("Reforming SimVascular OneDSolver results('model.in', 'output.vtp')")
        print("❌ Invalid output file. Expected '.vtp' extension.")
        print("Usage: convert_to_vtp('model.in', 'output.vtp')")
        return
    polydata = build_vtk_polydata(input_file)
    save_polydata_to_vtp(polydata, output_file)

# Example usage:
convert_to_vtp("12_AortoFem_Pulse_R.in", "12_AortoFem_Pulse_R.vtp")

