import time
import urllib.request
import logging
import sys
import json
import os

RETRIES = 50

def wait_for_server(port):
    for _ in range(RETRIES):
        try:
            fp = urllib.request.urlopen(f"http://localhost:{port}")
            fp.close()
            return
        except:
            time.sleep(2)
    exit(-1)

if __name__ == "__main__":
    model:str = os.getenv('PARSING_MODEL')
    assert model.lower() in ('spring', 'amrbart', 'amrlib'), f"Invalid AMR model name:{model}, should be in ['Spring', 'AMRBART', 'amrlib']"

    port=os.getenv('PORT')
    assert port, 'port must be set'

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    logging.info('waiting for server....')
    wait_for_server(port)
    logging.info('server connected')

    if model=='spring':
        parser.add_argument('--checkpoint', type=str, required=True,
            help="Required. Checkpoint to restore.")
        parser.add_argument('--model', type=str, default='facebook/bart-base',
            help="Model config to use to load the model class.")
        parser.add_argument('--penman-linearization', action='store_true',
            help="Predict using PENMAN linearization instead of ours.")
        parser.add_argument('--use-pointer-tokens', action='store_true')
        parser.add_argument('--amrs-folder', type=str, required=True)
        parser.add_argument('--config_path', type=str, default=None)
        args = parser.parse_args()

        
        model_name = urllib.parse.quote_plus(args.model)
        checkpoint = urllib.parse.quote_plus(args.checkpoint)
        amrs_folder = urllib.parse.quote_plus(args.amrs_folder)
        penman_linearization:int = 1 if args.penman_linearization else 0
        use_pointer_tokens:int = 1 if args.use_pointer_tokens else 0
        config_path = urllib.parse.quote_plus(args.config_path) if args.config_path is not None else None

        if config_path is None:
            fp = urllib.request.urlopen(f"http://localhost:{port}/init_model_and_tokenizer?model_name={model_name}&checkpoint={checkpoint}&amrs_folder={amrs_folder}&penman_linearization={penman_linearization}&use_pointer_tokens={use_pointer_tokens}")
        else:
            fp = urllib.request.urlopen(f"http://localhost:{port}/init_model_and_tokenizer?model_name={model_name}&checkpoint={checkpoint}&amrs_folder={amrs_folder}&penman_linearization={penman_linearization}&use_pointer_tokens={use_pointer_tokens}&config_path={config_path}")

        fp.close()

    elif model=='amrbart':
        parser.add_argument('--config_file', type=str, required=True,
            help="Required. config file path for initialize model and tokenizer")
        parser.add_argument('--data_dir', type=str, default='../documents',
            help="Required. Where documents will be saved")
        parser.add_argument('--output_dir', type=str, default='../graphs',
            help="Required. Where predicted AMR graphs will be saved")
        parser.add_argument('--model_cache_dir', type=str, required=True,
            help="Required. Where to store the pretrained models downloaded from huggingface.co")
        parser.add_argument('--data_cache_dir', type=str, required=True,
            help="Required. Where to store the cached dataset")
        args = parser.parse_args()
        args=vars(args)
        
        assert args['config_file'] and not args['config_file'].isspace(), 'configuration file path must be provide'
        
        with open(args['config_file']) as f:
            config = f.read()

        config = json.loads(config)
        
        config['data_dir'] = args['data_dir']
        config['output_dir'] = args['output_dir']
        config['cache_dir'] = args['model_cache_dir']
        config['data_cache_dir'] = args['data_cache_dir']
        
        with open(args['config_file'], 'w') as f:
            json.dump(config, f)

        # init model and tokenizer from json config
        input_path = urllib.parse.quote_plus(args['config_file'])
        fp = urllib.request.urlopen(f"http://localhost:{port}/init_model_and_tokenizer?config_file_path={input_path}")
        encodedContent = fp.read()
        logging.warning(encodedContent.decode("utf8"))
        fp.close()

    elif model=='amrlib':
        parser.add_argument('--model_dir', type=str, required=True,
            help="Required. Where model checkpoints are saved")
        parser.add_argument('--output_dir', type=str, default='../graphs',
            help="Where predicted AMR graphs will be saved")
        args = parser.parse_args()

        model_dir = urllib.parse.quote_plus(args.model_dir)
        output_dir = urllib.parse.quote_plus(args.output_dir)

        fp = urllib.request.urlopen(f"http://localhost:{port}/init_model_and_tokenizer?output_dir={output_dir}&model_dir={model_dir}")
        fp.close()