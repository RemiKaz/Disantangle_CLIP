import data
import utils
import argparse
import torch 
import clip
import csv
import os

if __name__ == "__main__":

    def parse_args():
        # Training settings
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "--root_data",
            type=str,
            default="/lustre/fsmisc/dataset/imagenet",
            help="Root path to data",
        )
        parser.add_argument(
            "--root_parts",
            type=str,
            default="/media/remi/RemiPro1/PartImageNetPP",
            help="Root path to parts data",
        )
        parser.add_argument(
            "--root_monumai",
            type=str,
            default="/media/remi/RemiPro1/VLG-CBM/DataRemi/OD-MonuMAI/MonuMAI_dataset",
            help="Root path to MonuMAI dataset",
        )
        parser.add_argument(
            "--root_cub",
            type=str,
            default="/media/remi/RemiPro1/VLG-CBM/datasets/CUB",
            help="Root path to CUB dataset",
        )
        parser.add_argument(
            "--root_coco",
            type=str,
            default="/lustre/fsmisc/dataset/COCO",
            help="Root path to COCO dataset",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="results",
            help="Directory to save result CSV files",
        )
        parser.add_argument(
            "--classes_to_probe",
            type=str,
            default="Plunger-plunger_handle",
            help="subset to probe",
        )
        parser.add_argument(
            "--clip_concepts",
            type=str,
            default='handle',
            help="concept text to use to compute CLIP scores",
        )
        parser.add_argument(
            "--model",
            default="ViT-B-16",
            type=str,
            metavar="MODEL",
            help="Name of model to use in [ViT-B-16;laion2b_s34b_b88k,ViT-H-14;laion2B-s32B-b79K]",
        )
        parser.add_argument("--pretrained", default="laion2b_s34b_b88k", type=str) 
        parser.add_argument(
            "--experiments_to_run",
            type=str,
            default='plot_clip_scores_histogram,',
            help="experiments to run in [save_sample,save_sample+mask,plot_segmentation_map,plot_clip_scores_histogram,print_sample_with_highest_clip_score, plot_diff_clip_scores_histogram]",
        )

        parser.add_argument(
            "--ids_samples_save",
            type=str,
            default='0',
            help="if save_sample, ids of samples to save",
        )      
        parser.add_argument(
            "--disantangled_version",
            type=str,
            default='median-LTC-full',
            help="if present, indicates the parameters of the disantangling [median/treshold - LTC/loc/register - random/full]",
        )          
        parser.add_argument(
            "--classifier_dataset",
            type=str,
            default="imagenet",
            help="The name of the classifier dataset [imagenet ,coco, monumai, cub]",
        )
        return parser.parse_args()

    # Load data/explainer/model
    args = parse_args()

    if args.disantangled_version == 'None':
        args.disantangled_version = False

    # Models 'ViT-L/14','ViT-B/32','ViT-L/14@336px'
    if args.classifier_dataset == 'imagenet':
        dataset = data.PIN_dataset(root_data=args.root_data,root_parts=args.root_parts, split='train', class_restrict=[args.classes_to_probe],
                                clip_concepts=[args.clip_concepts],model_name=args.model,pretrained=args.pretrained,
                                disantangled_version=args.disantangled_version)

    elif args.classifier_dataset == 'monumai':
        dataset = data.MonumaiDataset(root=args.root_monumai, phase='test', model_name=args.model, pretrained=args.pretrained,
                                   reduce_samples=False,device='cuda',clip_concepts=[args.clip_concepts],
                                   disantangled_version=args.disantangled_version,class_restrict=[args.classes_to_probe])

    elif args.classifier_dataset == 'coco':
        dataset = data.COCODataset(root_data=args.root_coco, root_parts=args.root_coco, split='val', pth_metadata=None,
                                   pth_metadata_parts=None, model_name=args.model, pretrained=args.pretrained,
                                   reduce_samples=False, train_cbm_mode=False, class_restrict=[args.classes_to_probe], clip_concepts=[args.clip_concepts]
                                   ,disantangled_version=args.disantangled_version)

    elif args.classifier_dataset == 'cub':
        dataset = data.CUBDataset(root=args.root_cub, phase='val', model_name=args.model, pretrained=args.pretrained,
                                   reduce_samples=False,device='cuda',clip_concepts=[args.clip_concepts],
                                   disantangled_version=args.disantangled_version,class_restrict=[args.classes_to_probe])

    exps = args.experiments_to_run.split(',')
    
    # Return sample
    
    if 'save_sample' in exps:
        for i in args.ids_samples_save.split(','):       
            dataset.save_image_id(int(i),wanted_concept=args.classes_to_probe.split('-')[-1],save_also_only_concept=True)

    if 'save_sample+mask' in exps:
        for i in args.ids_samples_save.split(','):
            dataset.save_image_id(int(i),wanted_concept=args.classes_to_probe.split('-')[-1],save_also_only_concept=True,save_also_mask=True)

    # Plot segmentation map

    if 'plot_segmentation_map' in exps:
        for i in args.ids_samples_save.split(','):
            data_sample = dataset.List_data_image[int(i)]
            utils.plot_segmentation_map(data_sample)

    # Stats clip scores

    if 'plot_clip_scores_histogram' in exps:
        '''utils.plot_clip_scores_histogram(dataset,dataset.clip_concepts,check_attribute=['all','present','absent','inpainted'],concept_to_check=args.classes_to_probe.split('-')[-1])'''
        dict_mean_scores = utils.plot_clip_scores_histogram(dataset,dataset.clip_concepts,check_attribute=['all','present','absent'],concept_to_check=args.classes_to_probe.split('-')[-1])
        # Define the CSV file name
        os.makedirs(args.output_dir, exist_ok=True)
        csv_file = os.path.join(args.output_dir, f"mean_scores_{args.disantangled_version}_{args.classifier_dataset}.csv")
        
        # Writing to csv file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not os.path.isfile(csv_file):
                writer.writerow(["Key", "Mean", "Variance"])
            # Write the data
            for key in dict_mean_scores.keys():
                if args.disantangled_version:
                    writer.writerow([f"{args.classes_to_probe}"+key+f"_{args.disantangled_version}", dict_mean_scores[key]['mean'], dict_mean_scores[key]['variance'],dict_mean_scores[key]['list_scores']])
                else:                    
                    writer.writerow([key, dict_mean_scores[key]['mean'], dict_mean_scores[key]['variance'],dict_mean_scores[key]['list_scores']])

        print(f"Data has been written to {csv_file}")
        
    if 'plot_diff_clip_scores_histogram' in exps:
        utils.plot_diff_clip_scores_histogram(dataset,dataset.clip_concepts,concept_to_check=args.classes_to_probe.split('-')[-1])

    # Print sample with highest clip score
    if 'print_sample_with_highest_clip_score' in exps:
        utils.print_sample_with_highest_clip_score(dataset,args.clip_concepts)

