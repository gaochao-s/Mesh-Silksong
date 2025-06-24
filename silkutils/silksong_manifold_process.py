import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from meshdata.mesh_io import load_mesh, write_obj
from meshdata.mesh_structure import Mesh
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='config for non-manifold mesh',  
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  
    )

    parser.add_argument('--input_file',
                        default='silkutils/demo_test/shapenetv2_03761084_ee5861.obj',
                       type=str, 
                       help='inputfile')
                       
    
    parser.add_argument('--quant_resolution',
                       default=1024,
                       type=int, 
                       help='quantization resolution')

    parser.add_argument('-o', '--output_dir',
                       default='silkutils/demo_test/manifold_repair',
                       help='output dir')

    parser.add_argument('--verbose',
                       action='store_true',
                       help='if show detail')

    parser.add_argument('--max_edge_graph_edges',
                       default=100,
                       type=int, 
                       help='structure limit')
    
    parser.add_argument('--max_nonmani_verts',
                       default=500,
                       type=int, 
                       help='structure limit2')
    
    parser.add_argument('--min_CC_face',
                       default=4,
                       type=int, 
                       help='structure limit3')
    
    parser.add_argument('--max_face_num',
                       default=16000,
                       type=int, 
                       help='structure limit4')
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    structure_limit_kwargs={
        'NM_max_edge_graph': args.max_edge_graph_edges,
        'NM_max_nonmani_verts': args.max_nonmani_verts,
        'min_CC_face': args.min_CC_face,
        'max_face_num_p': args.max_face_num,
    }

    output_dir=args.output_dir
    input_file=args.input_file
    resolution=args.quant_resolution
    verbose=args.verbose

    os.makedirs(output_dir, exist_ok=True)

    save_name=os.path.basename(input_file).split('.')[0]
    
    # M1: loaded by trimesh, normalized, pre clean by kiui
    M1_save_path=os.path.join(output_dir, f"M1_{save_name}.obj")
    # M2: mesh quantization, nonmani processing, colored by connected component
    M2_save_path=os.path.join(output_dir, f"M2_{save_name}.obj")

    try:
        vertices, faces = load_mesh(input_file, clean=True)
    except Exception as e:
        raise Exception('[E] loading Failed')
    
    
    write_obj(vertices, faces, M1_save_path)

    mesh = Mesh(vertices=vertices, triangles=faces, discrete_bins=resolution, verbose=verbose, debugging=False, non_mani_process=True, NM_max_edge_graph=structure_limit_kwargs["NM_max_edge_graph"], NM_max_nonmani_verts=structure_limit_kwargs["NM_max_nonmani_verts"], min_CC_face=structure_limit_kwargs["min_CC_face"], M2_path=M2_save_path, just_process=True) 
    



