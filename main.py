# main.py
# -*- encoding: utf-8 -*-

import argparse
import os
from datetime import datetime

from torch import optim
from denoise_bayes_vae.bayes_vae import BayesianVAE
from denoise_bayes_vae.dataset import SpeechDatasetLoader
from denoise_bayes_vae.test import denoise, load_model
from denoise_bayes_vae.train import train
from utils.audio_wav import save_wav
from utils.device import get_compute_device, init_gpu_cache
from utils.exception import get_exception_traceback
from utils.file_utils import FileUtil
from utils.logger import Logger

timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
Logger().configure(timestamp)
logger = Logger().get_logger()

def parse_arguments():
    """
    Parse command line
    :return:
    """
    parser = argparse.ArgumentParser(prog="bayesian_vae", description="Bayesian VAE")
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--ft', dest='fine_tuning', action='store_true')
    parser.add_argument('--autodiff', dest='autodiff', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--manifest', dest='manifest', action='store')
    parser.add_argument('--noisy_dir', dest='noisy_dir', action='store')
    parser.add_argument('--clean_dir', dest='clean_dir', action='store')
    parser.add_argument('--test_input', dest='test_input', action='store')
    parser.add_argument('--pdf', dest='pdf', default='gaussian', action='store')
    parser.add_argument('--df', dest='df', default=3.0, action='store')
    parser.add_argument('--output_dir', dest='output_dir', action='store')
    parser.add_argument('--train_val_ratio', dest='train_val_ratio', default=0.8, action='store')
    parser.add_argument('--input_dim', dest='input_dim', default=16000, action='store')
    parser.add_argument('--latent_dim', dest='latent_dim', default=128, action='store')
    parser.add_argument('--batch_size', dest='batch_size', default=16, action='store')
    parser.add_argument('--epoch', dest='epoch', default=10, action='store')
    parser.add_argument('--patience', dest='patience', default=3, action='store')
    parser.add_argument('--report', dest='report', action='store_true')

    return parser.parse_args()

def is_valid_manifest(manifest):
    """
    check whether manifest file is valid
    :param manifest: (dict)
    :return: (bool)
    """
    if 'pdf' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'pdf\' key in manifest.'.format(manifest))
        return False
    if 'df' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'df\' key in manifest.'.format(manifest))
        return False
    if 'model' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'model\' key in manifest.'.format(manifest))
        return False
    if 'train_val_ratio' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'train_val_ratio\' key in manifest.'.format(manifest))
        return False
    if 'noisy_dir' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'noisy_dir\' key in manifest.'.format(manifest))
        return False
    if 'clean_dir' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'clean_dir\' key in manifest.'.format(manifest))
        return False
    if 'batch_size' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'batch_size\' key in manifest.'.format(manifest))
        return False
    if 'epoch' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'epoch\' key in manifest.'.format(manifest))
        return False
    if 'patience' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'patience\' key in manifest.'.format(manifest))
        return False
    if 'input_dim' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'input_dim\' key in manifest.'.format(manifest))
        return False
    if 'latent_dim' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'latent_dim\' key in manifest.'.format(manifest))
        return False
    if 'autodiff' not in manifest:
        logger.error('Invalid manifest file {}. Not found \'autodiff\' key in manifest.'.format(manifest))
        return False
    if not os.path.isfile(manifest['model']):
        logger.error('Not found model file {}.'.format(manifest['model']))
        return False

    return True

def main():
    args = parse_arguments()
    fine_tuning = False
    noisy_dir = args.noisy_dir
    clean_dir = args.clean_dir
    autodiff = args.autodiff
    dataset = None
    input_dim = 0
    latent_dim = 0
    epoch = 0
    batch_size = 0
    df = 0.0
    train_val_ratio = 0.0
    patience = 0
    manifest = None
    is_report_provided = False
    output_dir_entry = None

    if args.train and args.test:
        logger.error('You must input either --train or --test.')
        exit(1)

    elif args.train:
        if args.fine_tuning:
            output_dir_entry = f"train_ft_{timestamp}"
        else:
            output_dir_entry = f"train_{timestamp}"

    elif args.test:
        output_dir_entry = f"test_{timestamp}"

    else:
        logger.error('You must input either --train or --test.')
        exit(1)

    # if args.output_dir is not specified, output file is saved in current directory
    output_root_dir = os.path.join(os.getcwd(), 'output')

    if args.output_dir:
        if os.path.isdir(args.output_dir):
            output_root_dir = args.output_dir
        else:
            logger.error('Not found {0} directory for \'--output_dir {0}.\'.'.format(args.output_dir))

    output_dir = os.path.join(output_root_dir, output_dir_entry)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        input_dim = int(args.input_dim)
    except ValueError:
        logger.error('Invalid value --input_dim {}. You must input integer value.'.format(args.input_dim))
        exit(1)

    try:
        latent_dim = int(args.latent_dim)
    except ValueError:
        logger.error('Invalid value --latent_dim {}. You must input integer value.'.format(args.latent_dim))
        exit(1)

    pdf = args.pdf

    try:
        df = float(args.df)
    except ValueError:
        logger.error('Invalid value --df {}. You must input float value.'.format(args.df))
        exit(1)

    try:
        train_val_ratio = float(args.train_val_ratio)
    except ValueError:
        logger.error('Invalid value --train_val_ratio {}. '
                     'You must input a float value greater than 0 and less than 1.'.format(args.train_val_ratio))
        exit(1)

    try:
        batch_size = int(args.batch_size)
    except ValueError:
        logger.error('Invalid value --batch_size {}. You must input integer value.'.format(args.batch_size))
        exit(1)

    try:
        epoch = int(args.epoch)
    except ValueError:
        logger.error('Invalid value --epoch {}. You must input integer value.'.format(args.epoch))
        exit(1)

    try:
        patience = int(args.patience)
    except ValueError:
        logger.error('Invalid value --patience {}. You must input integer value.'.format(args.patience))
        exit(1)

    device = get_compute_device()
    init_gpu_cache()

    ''' training mode '''
    if args.train:
        if not args.noisy_dir:
            logger.error('You must input --noisy_dir [noisy data directory].')
            exit(1)

        if not os.path.isdir(args.noisy_dir):
            logger.error('Not found {} directory for \'--noisy_dir\'.'.format(args.noisy_dir))
            exit(1)

        if not args.clean_dir:
            logger.error('You must input --clean_dir [clean data directory].')
            exit(1)

        if not os.path.isdir(args.clean_dir):
            logger.error('Not found {} directory for \'--clean_dir\'.'.format(args.clean_dir))
            exit(1)

        if args.fine_tuning: # fine-tuning mode
            if not args.manifest:
                logger.error('You must input --manifest [manifest file path to load].')
                exit(1)

            try:
                manifest = FileUtil.from_json_file(args.manifest)
            except Exception as exc:
                logger.error('Exception in from_json_file(). Caused by {}.'
                      .format(args.manifest, get_exception_traceback(exc)))
                exit(1)

            fine_tuning = True
            model_file_path = manifest['model']
            input_dim = manifest['input_dim']
            latent_dim = manifest['latent_dim']
            pdf = manifest['pdf']
            df = manifest['df']
            train_val_ratio = manifest['train_val_ratio']
            batch_size = manifest['batch_size']
            epoch = manifest['epoch']
            patience = manifest['patience']
            model = BayesianVAE(input_dim, latent_dim, pdf, df).to(device)

            # load pre-trained model
            if not os.path.isfile(model_file_path):
                logger.error('Not found model file {}.'.format(model_file_path))
                exit(1)
            load_model(model, model_file_path, device)
            learning_rate = 3e-3

            # set optimizer for fine-tuning
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        else: # pre-training mode
            model = BayesianVAE(input_dim, latent_dim, pdf, df).to(device)

            # set optimizer for pre-tuning
            learning_rate = 1e-4
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # set model, manifest, report filename
        model_file_path = os.path.join(output_dir, '{}.pth'.format(timestamp))
        manifest_file_path = os.path.join(output_dir, '{}.manifest'.format(timestamp))

        if args.report: # collapse signal report
            is_report_provided = True

        report_path_prefix = os.path.join(output_dir, '{}_'.format(timestamp))

        logger.info(f'\n'
                    f'**** Bayesian VAE training ****\n'
                    f'start = {datetime.now()}\n'
                    f'fine tuning (--ft) = {fine_tuning}\n'
                    f'compute device = {device}\n'
                    f'probability density function (--pdf) = {pdf}\n'
                    f'degree of freedom (--df) = {df}\n'
                    f'model file = {model_file_path}\n'                    
                    f'manifest file = {manifest_file_path}\n'
                    f'autodiff (--autodiff) = {autodiff}\n'
                    f'train/validation data ratio (--train_val_ratio) = {train_val_ratio}\n'
                    f'noisy_dir (--noisy_dir) = {noisy_dir}\n'
                    f'clean_dir (--clean_dir) = {clean_dir}\n'
                    f'batch_size (--batch_size) = {batch_size}\n'
                    f'epoch (--epoch) = {epoch}\n'
                    f'patience (--patience) = {patience}\n'
                    f'input dim (--input_dim) = {input_dim}\n'
                    f'latent dim (--latent_dim) = {latent_dim}\n'
                    f'learning rate = {learning_rate}\n')

        start = datetime.now()
        manifest = {
            "start": start.strftime("%Y%m%dT%H%M%S"),
            "end": None,
            "fine_tuning": fine_tuning,
            "pdf": pdf,
            "df": df,
            "model": model_file_path,
            "train_val_ratio": train_val_ratio,
            "noisy_dir": noisy_dir,
            "clean_dir": clean_dir,
            "batch_size": batch_size,
            "epoch": epoch,
            "patience": patience,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "autodiff": autodiff,
            "learning_rate": learning_rate
        }

        FileUtil.to_json_file(manifest, manifest_file_path)

        try:
            dataset = SpeechDatasetLoader(noisy_dir,
                                          clean_dir,
                                          train_val_ratio,
                                          batch_size)
        except Exception as exc:
            print('Exception in SpeechDatasetLoader. Caused by {}.', get_exception_traceback(exc))
            exit(1)

        train_dataloader = dataset.get_train_dataloader()
        val_dataloader = dataset.get_val_dataloader()

        ''' training '''
        try:
            train(model,
                  optimizer,
                  train_dataloader,
                  val_dataloader,
                  device,
                  model_file_path,
                  autodiff,
                  epoch,
                  patience,
                  report_path_prefix,
                  is_report_provided)
        except Exception as exc:
            print('Exception in train. Caused by {}.', get_exception_traceback(exc))
            exit(1)

        manifest["end"] = datetime.now().strftime("%Y%m%dT%H%M%S")
        FileUtil.to_json_file(manifest, manifest_file_path)

    ''' inference mode '''
    if args.test:
        if not args.manifest:
            print('You must input --manifest [manifest file path to load].')
            exit(1)

        if not os.path.isfile(args.manifest):
            print('Not found {0} file for \'--manifest {0}.\'.'.format(args.manifest))
            exit(1)

        if not args.test_input:
            print('You must input --test_input [input file path to denoise].')
            exit(1)

        if not os.path.isfile(args.test_input):
            print('Not found {0} file for \'--test_input {0}.\'.'.format(args.test_input_file))

        ''' parse manifest file for bayesian vae model '''
        manifest = None
        try:
            manifest = FileUtil.from_json_file(args.manifest)
        except Exception as exc:
            print('Exception in from_json_file(). Caused by {}.'.format(get_exception_traceback(exc)))
            exit(1)

        if not is_valid_manifest(manifest):
            exit(1)

        pdf = manifest['pdf']
        df = manifest['df']
        model_file_path = manifest['model']
        input_dim = manifest['input_dim']
        latent_dim = manifest['latent_dim']
        fine_tuning = manifest['fine_tuning']
        autodiff = manifest['autodiff']
        learning_rate = manifest['learning_rate']
        input_filename = os.path.splitext(os.path.basename(args.test_input))[0]
        model_filename = os.path.splitext(os.path.basename(model_file_path))[0]
        output_file_path = os.path.join(output_dir, 'rc_{}_{}_{}.wav'
                                        .format(input_filename, model_filename, timestamp))

        ''' setup parameters '''
        model = BayesianVAE(input_dim, latent_dim, pdf, df).to(device)
        load_model(model, model_file_path, device)

        print('**** Bayesian VAE inference ****')
        print('start = {}'.format(datetime.now()))
        print('compute device = {}'.format(device))
        print('model file = {}'.format(model_file_path))
        print('fine_tuning = {}'.format(fine_tuning))
        print('autodiff = {}'.format(autodiff))
        print('probability density function = {}'.format(pdf))
        if pdf == 'student_t':
            print('degree of freedom = {}'.format(df))
        print('test input file (--test_input) = {}'.format(args.test_input))
        print('output directory (--output_dir) = {}.'.format(output_dir))
        print('output file (denoised wav) = {}'.format(output_file_path))
        print('input dim = {}'.format(input_dim))
        print('latent dim = {}'.format(latent_dim))

        logger.info(f'\n'
                    f'**** Bayesian VAE training ****\n'
                    f'start = {datetime.now()}\n'
                    f'fine tuning (--ft) = {fine_tuning}\n'
                    f'compute device = {device}\n'
                    f'probability density function (--pdf) = {pdf}\n'
                    f'degree of freedom (--df) = {df}\n'
                    f'model file = {model_file_path}\n'                    
                    f'manifest file = {manifest}\n'
                    f'autodiff (--autodiff) = {autodiff}\n'
                    f'train/validation data ratio (--train_val_ratio) = {train_val_ratio}\n'
                    f'noisy_dir (--noisy_dir) = {noisy_dir}\n'
                    f'clean_dir (--clean_dir) = {clean_dir}\n'
                    f'batch_size (--batch_size) = {batch_size}\n'
                    f'epoch (--epoch) = {epoch}\n'
                    f'patience (--patience) = {patience}\n'
                    f'input dim (--input_dim) = {input_dim}\n'
                    f'latent dim (--latent_dim) = {latent_dim}\n'
                    f'learning rate = {learning_rate}\n')

        ''' inference '''
        denoised_waveform, sampling_rate = denoise(model, args.test_input, device)

        ''' save wav file '''
        save_wav(denoised_waveform, output_file_path, sample_rate=sampling_rate)

if __name__ == "__main__":
    main()
