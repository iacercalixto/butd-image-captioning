from utils import create_input_files, create_scene_graph_input_files
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('prepare')
    parser.add_argument('-s', dest='sg', action='store_true', help='if we need to prepare scene graph files')
    parser.add_argument('--dataset', default='coco', type=str, help='name of dataset')
    parser.add_argument('--dataset_path', default='data/caption_datasets/dataset_coco.json', type=str,
                        help='path to the chosen dataset')
    args = parser.parse_args()
    # Create input files (along with word map)
    create_input_files(dataset=args.dataset,
                       karpathy_json_path=args.dataset_path,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='final_dataset',
                       max_len=50)
    if args.sg:
        create_scene_graph_input_files(dataset=args.dataset,
                                       karpathy_json_path=args.dataset_path,
                                       output_folder='final_dataset')