from pathlib import Path

import torch

from spring_amr.penman import encode
from spring_amr.utils import instantiate_model_and_tokenizer

import os
import logging
from typing import Optional, List
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

KEEP_RUNNING = True
AMRS_FOLDER = ''
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _setup_model_and_tokenizer(
    checkpoint: str,
    amrs_folder: str,
    model_name: Optional[str] = None,
    penman_linearization: Optional[bool] = None,
    use_pointer_tokens: Optional[bool] = None,
):
    global AMRS_FOLDER
    global model
    global tokenizer
    AMRS_FOLDER = amrs_folder

    model, tokenizer = instantiate_model_and_tokenizer(
        model_name,
        dropout=0.,
        attention_dropout=0,
        penman_linearization=penman_linearization,
        use_pointer_tokens=use_pointer_tokens,
    )
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    model.to(device)
    model.eval()

def keep_running():
    return KEEP_RUNNING

def shut_down_server():
    global KEEP_RUNNING
    KEEP_RUNNING = False

def is_model_and_tokenizer_init():
    return model is not None and tokenizer is not None

def read_file_in_batches(path, batch_size=1000, max_length=100):

    data = []
    idx = 0
    for line in Path(path).read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        n = len(line.split())
        if n > max_length:
            continue
        data.append((idx, line, n))
        idx += 1

    def _iterator(data):

        data = sorted(data, key=lambda x: x[2], reverse=True)

        maxn = 0
        batch = []

        for sample in data:
            idx, line, n = sample
            if n > batch_size:
                if batch:
                    yield batch
                    maxn = 0
                    batch = []
                yield [sample]
            else:
                curr_batch_size = maxn * len(batch)
                cand_batch_size = max(maxn, n) * (len(batch) + 1)

                if 0 < curr_batch_size <= batch_size and cand_batch_size > batch_size:
                    yield batch
                    maxn = 0
                    batch = []
                maxn = max(maxn, n)
                batch.append(sample)

        if batch:
            yield batch

    return _iterator(data), len(data)

def predict_amrs_from_plaintext(
        texts: List[str], 
        batch_size: Optional[int]=None, 
        beam_size: Optional[int]=None,
        restore_name_ops: Optional[bool]=None, 
        only_ok: Optional[bool]=None, 
        output_path: Optional[str]=None,
        debug_info: bool=False,
    ):

    batch_size = batch_size if batch_size is not None else 1000
    beam_size = beam_size if beam_size is not None else 1
    restore_name_ops = restore_name_ops if restore_name_ops is not None else False
    only_ok = only_ok if only_ok is not None else False


    for path in texts:

        iterator, nsent = read_file_in_batches(path, batch_size)

        for batch in iterator:
            if not batch:
                continue
            ids, sentences, _ = zip(*batch)
            x, _ = tokenizer.batch_encode_sentences(sentences, device=device)
            with torch.no_grad():
                model.amr_mode = True

                out = model.generate(**x, max_length=512, decoder_start_token_id=0, num_beams=beam_size)

            bgraphs = []
            for idx, sent, tokk in zip(ids, sentences, out):

                graph, status, (lin, backr) = tokenizer.decode_amr(tokk.tolist(), restore_name_ops=restore_name_ops)

                if only_ok and ('OK' not in str(status)):
                    continue
                graph.metadata['status'] = str(status)
                graph.metadata['source'] = path
                graph.metadata['nsent'] = str(idx)
                graph.metadata['snt'] = sent
                bgraphs.append((idx, graph))

            final_penman_graphs = []
            for i, g in bgraphs:
                final_penman_graphs.append(encode(g))
                
            if output_path is not None:
                with open(output_path, 'w') as out_file:
                    out_file.write('\n'.join(final_penman_graphs))

        break

class RequestHandler(BaseHTTPRequestHandler):
    def _set_response(self, exit_code: int=200):
        self.send_response(exit_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        url_parsed = urlparse(self.path)
        queries: dict = parse_qs(url_parsed.query)
        
        output = queries.get('output', [0])
        output = int(output[0]) > 0
        output_path = ''
        
        is_shutting_down = queries.get('shut_down_server', [0])
        is_shutting_down = int(is_shutting_down[0]) > 0
        if is_shutting_down:
            shut_down_server()
        
        if 'init_model_and_tokenizer' in url_parsed.path:
            model_name = queries.get('model_name', [])
            checkpoint = queries.get('checkpoint', [])
            amrs_folder = queries.get('amrs_folder', [])
            penman_linearization = queries.get('penman_linearization', [0])
            use_pointer_tokens = queries.get('use_pointer_tokens', [0])

            model_name = model_name[0] if len(model_name) > 0 else None
            checkpoint = checkpoint[0] if len(checkpoint) > 0 else None
            amrs_folder = amrs_folder[0] if len(amrs_folder) > 0 else None
            penman_linearization = int(penman_linearization[0]) > 0
            use_pointer_tokens = int(use_pointer_tokens[0]) > 0

            if len(checkpoint) == 0 or len(amrs_folder) == 0:
                self._set_response()
                self.wfile.write(f"error: model's checkpoint and the folder where graphs will be saved can't be None".encode('utf-8'))
                return

            _setup_model_and_tokenizer(
                model_name=model_name, 
                checkpoint=checkpoint, 
                amrs_folder=amrs_folder, 
                penman_linearization=penman_linearization, 
                use_pointer_tokens=use_pointer_tokens,
                )
        
        if 'predict_amrs_from_plaintext' in url_parsed.path:
            if not is_model_and_tokenizer_init():
                self._set_response()
                self.wfile.write(f"error: model and tokenizer needs to be inizialized before use".encode('utf-8'))
                return

            input_path = queries.get('input_path', [])
            batch_size = queries.get('batch_size', [])
            beam_size = queries.get('beam_size', [])
            restore_name_ops = queries.get('restore_name_ops', [0])
            only_ok = queries.get('only_ok', [0])
            output_file_name = queries.get('output_file_name', [])

            input_path = input_path[0] if len(input_path) > 0 else None
            batch_size = batch_size[0] if len(batch_size) > 0 else None
            beam_size = beam_size[0] if len(beam_size) > 0 else None
            restore_name_ops = int(restore_name_ops[0]) > 0
            only_ok = int(only_ok[0]) > 0

            if input_path is None:
                self._set_response()
                self.wfile.write(f"error: input path is required to predict the amr graph of the text into it".encode('utf-8'))
                return

            basename = os.path.basename(input_path) if len(output_file_name) == 0 else os.path.basename(output_file_name[0])
            basename = os.path.splitext(basename)[0]
            
            output_path = ''.join([AMRS_FOLDER, '/' , basename, '.amr.txt'])

            try:
                predict_amrs_from_plaintext(
                    texts=[input_path],
                    batch_size=batch_size,
                    beam_size=beam_size,
                    restore_name_ops=restore_name_ops,
                    only_ok=only_ok,
                    output_path=output_path,
                )
            except Exception as e:
                self._set_response()
                logging.error(e)
                self.wfile.write(f"e".encode('utf-8'))
                return
        self._set_response()
        self.wfile.write(f"{output_path}".encode('utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    logging.basicConfig(level=logging.ERROR)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    while keep_running():
        try:
            httpd.handle_request()
        except Exception:
            break
    httpd.server_close()

if __name__ == '__main__':
    run(port=1234)

