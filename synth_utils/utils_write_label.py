

def export_json_output_custom(dict_mask_category, pth_save_json_output):
    dict_via_region_data = {}
    for fn, ls_polygon in dict_mask_category.items():
        dict_via_region_data[fn+str(123)] = json_one_file(fn, ls_polygon, size=123)
    write_file(dict_via_region_data, pth_save_json_output)


def write_file(dict_via_region_data, pth_save_json_output):
    with open(pth_save_json_output / "via_region_data.json", "w") as file:
        file.write(str(dict_via_region_data).replace("'", '"'))


def json_one_file(filename, ls_polygon, size=None):
    print(ls_polygon)
    one_file = dict()
    one_file["filename"] = filename
    one_file["size"] = size
    one_file["regions"] = []
    ls_region = {}
    for ind, val in enumerate(ls_polygon):
        dict_json_polyline = dict()
        dict_json_polyline["name"] = "polygon"
        x, y = val[0].exterior.xy
        dict_json_polyline["all_points_x"] = list(map(int, list(x)))
        dict_json_polyline["all_points_y"] = list(map(int, list(y)))
        ls_region["shape_attributes"] = dict_json_polyline
        ls_region["region_attributes"] = {"class": {val[2]: "true"}}
        one_file["regions"].append(ls_region)
        ls_region = {}
    one_file["file_attributes"] = {}
    return one_file




