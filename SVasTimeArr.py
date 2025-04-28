import vtk
from collections import defaultdict

def SVas_TimeArr(vtp_file):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()
    polydata = reader.GetOutput()

    field_time_map = defaultdict(dict)
    num_arrays = polydata.GetPointData().GetNumberOfArrays()
    for i in range(num_arrays):
        arr = polydata.GetPointData().GetArray(i)
        if arr:
            name = arr.GetName()
            if '_t' in name:
                base, suffix = name.rsplit('_t', 1)
                if suffix.isdigit():
                    field_time_map[base][int(suffix)] = arr

    time_tags = sorted({k for v in field_time_map.values() for k in v})
    fields = sorted(set(field_time_map.keys()))

    data_list = []
    for time_step in time_tags:
        field_arrays = {}
        for field in fields:
            arr_name = f"{field}_t{time_step}"
            arr = polydata.GetPointData().GetArray(arr_name)
            if arr:
                field_arrays[field] = arr
        data_list.append((time_step, field_arrays))

    return data_list, polydata, field_time_map

