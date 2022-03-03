Chạy :
    + conda create -n synth_mrcnn python=3.6
    + conda activate synth_mrcnn
    + conda install -c conda-forge shapely
    + Cài file requirement.txt đính kèm theo thư mục này

Sinh data:
    python synth_main.py --input_dir datatset/input --output_dir datatset/output --count 5
    + --input_dir : Thư mục chứa ảnh backgrounds và foregrounds
    + --output_dir : Thư mục chứa ảnh sau khi sinh
    + --count : Số lượng ảnh sinh


Highlight ảnh sau khi Sinh:
    python synth_highlight.py --input_dir datatset/output/images --output_dir datatset/output/highlight
    + --input_dir : Thư mục chứa ảnh kèm file label
    + --output_dir : Thư mục chứa ảnh sau khi highlight, người dùng tự tạo
    + --count : Số lượng ảnh cần highlight