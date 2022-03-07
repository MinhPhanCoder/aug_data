from typing import List, Dict


def export_json_output_custom(dict_full_polygon_op: Dict, path_save_json_output: str):
    dict_via_region_data = {}
    for fn, ls_polygon in dict_full_polygon_op.items():
        dict_via_region_data[fn+str(123)] = op_one_image(fn, ls_polygon, size=123)
    write_file(dict_via_region_data, path_save_json_output)


def write_file(dict_via_region_data: Dict, path_save_json_output: str):
    with open(path_save_json_output / "via_region_data.json", "w") as file:
        file.write(str(dict_via_region_data).replace("'", '"'))


def op_one_image(filename: str, ls_polygon: List, size=None) -> Dict:
    regions = []
    for (polygon, _, category) in ls_polygon:
        x, y = polygon.exterior.xy
        shape_attribute = {
            "name": "polygon",
            "all_points_x": list(map(int, list(x))),
            "all_points_y": list(map(int, list(y)))
        }
        region_attribute = {
            "class": {
                category: "true"
            }
        }
        regions.append({
            "shape_attributes": shape_attribute,
            "region_attributes": region_attribute
        })
    op1img = {
        "filename": filename,
        "size": size,
        "regions": regions,
        "file_attributes": {}
    }
    return op1img




