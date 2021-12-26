import yaml
import argparse
from libs import Performer
def str2bool(str):
    return True if str.lower() == 'true' else False

def get_parser():
    # Priority : command line > config > default

    parser = argparse.ArgumentParser(description="code")

    # input: data path
    parser.add_argument("--data_type",                       default="single",                                          help="all/single/distribution data"                         )
    parser.add_argument("--data_root",                       default="jiangxingxun/Data/DFEW/data_affine/single_label/", help="path to personal Code and Data, withour data ori path")
    parser.add_argument("--work_dir",                        default="./work_dir/",                                  help="+YmdHMS, the folder for storing model and results"    )
    parser.add_argument("--fold_idx",       type=int,        default=1,                                              help="No.fold of data split for cross-validation"           )
    parser.add_argument("--saved_weights",                   default="None")
    

    # output: results write
    parser.add_argument("--tensorboard",     type=str2bool,  default=False)
    parser.add_argument("--txt_log",         type=str2bool,  default=True)
    parser.add_argument("--txt_log_toScreen",type=str2bool,  default=True)


    # speed: GPU and memory
    parser.add_argument("--gpu_id",                          default="7",                          help="the id of gpu server"                                                   )
    parser.add_argument("--num_workers",                     default=4                                                                                                           )
    parser.add_argument("--pin_memory",       type=str2bool, default=True                                                                                                       )
    parser.add_argument("--Flag_mGPU_blocks", type=str2bool, default=False                                                                                                        )
    parser.add_argument("--List_mGPU_blocks",                default=[0,1])
    parser.add_argument("--torch_optimize",   type=str2bool, default=True                                                                                                        )


    # hyper-parameters
    parser.add_argument("--config",                          default="./config/r3d_18.yaml",       help="path to the config file"                                                )

    parser.add_argument("--num_classes",                     default=7)
    parser.add_argument("--num_epoch",      type=int,        default=50,                           help="stop training from which epoch"                                         )
    parser.add_argument("--batch_size",     type=int,        default=32,                           help="batch size of training"                                                 )

    parser.add_argument("--test_gap",                        default="epoch"                                                                                                     )
    parser.add_argument("--test_stepLength",                 default=100                                                                                                         )   

    parser.add_argument("--optimizer",                       default="Adam",                       help="optimizer type"                                                         )
    parser.add_argument("--lr_init",        type=float,      default=1e-3,                         help="initial learning rate"                                                  )
    parser.add_argument("--lr_strategy",    type=str2bool,   default=True                                                                                                        )
    parser.add_argument("--CBL_beta",       type=float,      default=0.9999)

    # preprocess: data augment
    parser.add_argument("--train_data_augment",        type=str2bool,   default=False                                                                                                       )
    parser.add_argument("--Flag_RandomRotation",       type=str2bool,   default=True,                                                                                                       )
    parser.add_argument("--degree_RandomRotation",                      default=10,                           help="±10 degree"                                                             )
    parser.add_argument("--Flag_CenterCrop",           type=str2bool,   default=True                                                                                                        )
    parser.add_argument("--size_CenterCrop",                            default=224                                                                                                         )
    parser.add_argument("--Flag_RandomResizedCrop",    type=str2bool,   default=True                                                                                                        )
    parser.add_argument("--size_RandomResizedCrop",                     default=(224,224)                                                                                                   )
    parser.add_argument("--Flag_RandomHorizontalFlip", type=str2bool,   default=True                                                                                                        )
    parser.add_argument("--prob_RandomHorizontalFlip",                  default=0.5                                                                                                         )
    parser.add_argument("--Flag_RandomVerticalFlip",   type=str2bool,   default=True                                                                                                        )
    parser.add_argument("--prob_RandomVerticalFlip",                    default=0.5                                                                                                         )
    parser.add_argument("--Flag_RamdomErasing",        type=str2bool,   default=True                                                                                                        ) 
    parser.add_argument("--prob_RandomErasing",                         default=0.5,                                                                                                        )
    parser.add_argument("--Flag_Resize_te",            type=str2bool,   default=True                                                                                                        )
    parser.add_argument("--size_Resize_te",                             default=224                                                                                                         )
    parser.add_argument("--nframe",                    type=int,        default=16                                                                                                )
    parser.add_argument("--isconsecutive",             type=str2bool,   default=False                                                                                              )


    # model: model info, training strategy, 
    parser.add_argument("--model_name",                       default="r3d_18"                                                                                                  )
    parser.add_argument("--model_pretrain",    type=str2bool, default=True                                                                                                        )
    parser.add_argument("--pretrained_weights",               default="ImageNet"                                                                                                  )
    parser.add_argument("--model_init",        type=str2bool, default=True                                                                                                        )
    parser.add_argument("--y_start_from_zero", type=str2bool, default=False                                                                                                       )

                                   
    # loss: 
    parser.add_argument("--loss_type",         type=str,      default="CEL")
    parser.add_argument("--alpha_center_loss", type=float,    default=1e-3)

    return parser



if __name__ == "__main__":
    parser = get_parser()

    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.load(f)
        key = vars(p).keys()
        for k in default_args.keys():
            if k not in key:
                print("Wrong Arg:{}".format(k))
                assert (k in key)
        parser.set_defaults(**default_args)

    args      = parser.parse_args()
    performer = Performer.Performer(args)
    if args.test_gap == "epoch":
        performer.start_train_epochTest()
