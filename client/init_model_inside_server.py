import time
import urllib.request
import logging
logging.basicConfig(level=logging.INFO)

RETRIES = 50

def wait_for_server():
    for _ in range(RETRIES):
        try:
            fp = urllib.request.urlopen("http://localhost:1234")
            fp.close()
            return
        except:
            time.sleep(2)

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--amrs-folder', type=str, required=True)
    args = parser.parse_args()

    logging.info('waiting for server....')
    wait_for_server()
    
    model_name = urllib.parse.quote_plus(args.model)
    checkpoint = urllib.parse.quote_plus(args.checkpoint)
    amrs_folder = urllib.parse.quote_plus(args.amrs_folder)
    penman_linearization:int = 1 if args.penman_linearization else 0
    use_pointer_tokens:int = 1 if args.use_pointer_tokens else 0
    fp = urllib.request.urlopen(f"http://localhost:1234/init_model_and_tokenizer?checkpoint={checkpoint}&amrs_folder={amrs_folder}&penman_linearization={penman_linearization}&use_pointer_tokens={use_pointer_tokens}")
    fp.close()