import time, os
import argparse
from src.utils import color_dict_normal, get_model, LOG_DIR, get_logger, get_dataset_config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from src.utils.utils import download_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MIDX')
    parser.add_argument('--model', '-m', type=str, default='TransE', help='model name')
    parser.add_argument('--dataset', '-d', type=str, default='fb15k', help='dataset name')
    parser.add_argument('--log_path', '-p', type=str, default='./log', help='log path to save files')
    parser.add_argument('--mode', type=str, default='light', help='whether to enable nni')
    args, command_line_args = parser.parse_known_args()
    model_class, model_conf = get_model(args.model)
    parser = model_class.add_model_specific_args(parser)
    args_ = parser.parse_args(command_line_args)
    for k, v in vars(args_).items():
        for arg in command_line_args:
            if k in arg:
                model_conf[k] = v
                break
    model_conf['mode'] = args.mode
    if args.mode == 'tune':
        # hypo_params = nni.get_next_parameter()
        # model_conf.update(hypo_params)
        pass

    log_path = time.strftime(f"{model_class.__name__}-{args.dataset}-{args_.sampler}-{model_conf['num_neg']}-%Y-%m-%d-%H-%M-%S.log", \
        time.localtime())
    console_logger = get_logger(args.log_path,log_path)
    # tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name="tensorboard/" + log_path)
    tb_logger = TensorBoardLogger(save_dir=os.path.join(args.log_path,  log_path), name="tensorboard")

    dataset_class = model_class.get_dataset_class()
    dataset_conf = get_dataset_config(args.dataset)
    
    import os
    if not os.path.exists(os.path.join("./data", args.dataset)):
        download_dataset(dataset_conf['url'], args.dataset)

    if model_conf['fix_seed']:
        seed_everything(model_conf['seed'])

    dataset = dataset_class(name=args.dataset, config=dataset_conf)
    trn, val, tst = dataset.build()

    model = model_class(model_conf, trn)

    if args.mode == 'tune':
        show_progress_bar = False
    else:
        show_progress_bar = True

    # print("CUDA:", os.environ['CUDA_VISIBLE_DEVICES'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    trainer = Trainer(
        accelerator='gpu', 
        devices=model_conf['gpu'],
        # auto_select_gpus=True,
        logger=tb_logger,
        max_epochs=model_conf['epochs'],
        enable_progress_bar=show_progress_bar
    )

    console_logger.info('\n' + color_dict_normal(model_conf, False))

    trn_loader = trn.train_loader(batch_size=model_conf['batch_size'], num_workers=model_conf['num_workers'])
    val_loader = val.eval_loader(batch_size=model_conf['eval_batch_size'], num_workers=model_conf['num_workers'])
    trainer.fit(model, trn_loader, val_loader,)

    tst_loader = tst.eval_loader(batch_size=model_conf['eval_batch_size'], num_workers=model_conf['num_workers'])
    trainer.test(model, tst_loader)
