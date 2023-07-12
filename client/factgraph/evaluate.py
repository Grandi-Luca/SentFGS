from factgraph.process_utils import simplify_amr_nopar
import json
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from factgraph.utils_evaluate import preprocess_function
import pandas as pd
import logging
import sys
from amr_utils.amr_readers import AMR_Reader

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_amr(data,summ_location):
        reader = AMR_Reader()
        summ_amr_list = reader.load(summ_location, remove_wiki=True)
        #all summary amr        
        dict_summ_amr_data = {}
        for amr in summ_amr_list:
            dict_summ_amr_data[amr.metadata['snt']] = amr

        error_log_claim = 0
        for d in data:
            try:
                # s = " ".join(d['summary'].split())
                s = d['summary']
                graph = dict_summ_amr_data[s].graph_string()
                graph_simple, triples = simplify_amr_nopar(graph)
                d['graph_summary'] = {}
                graph_simple = ' '.join(graph_simple)
                d['graph_summary']['amr_simple'] = graph_simple
                d['graph_summary']['triples'] = json.dumps(triples)

            except:
                error_log_claim += 1
                print("skip graph claim", error_log_claim)
                d['graph_summary'] = {}
                d['graph_summary']['amr_simple'] = ''
                d['graph_summary']['triples'] = ''

        return data

def test(model, test_data,num_document_graphs):
    dataloader_test = DataLoader(test_data, batch_size=20, collate_fn=collate_fn)
    pred = eval_factgraph(model, dataloader_test,num_document_graphs)
    return pred

def collate_fn(batch):
    """
    data: is a list of tuples with (example, label, length)
            where "example" is a tensor of arbitrary shape
            and label/length are scalars
    """
    data = {}
    for key in batch[0].keys():
        data[key] = [item[key] for item in batch]

    ignore_fields = {'edge_index', 'edge_type', 'summary', 'article', 'head_nodes_graph',
                    'tail_nodes_graph', 'head_words', 'tail_words'}
    for key in data.keys():
        if key not in ignore_fields:
            data[key] = torch.tensor(data[key], dtype=torch.long).cuda()

    return data

def eval_factgraph(model, dataloader_val,num_document_graphs):
    model.eval()
    with torch.no_grad():
        for step, input in enumerate(dataloader_val):
            graph_structure = {"num_graphs": num_document_graphs + 1, "mask_graph": input["mask_graph"],
                            "edge_index": input["edge_index"], "edge_type": input["edge_type"]}
            outputs = model(input["input_ids"], input["attention_mask"], input["graphs"],
                            input["graphs_attn"], graph_structure)
            preds = outputs

            # preds = torch.sigmoid(preds)

            pred_labels = {}
            count = {}
            for i,j in zip(input["hyp_id"],preds):
                if i.item() in pred_labels:
                    pred_labels[i.item()] += j[1].item()
                    count[i.item()] += 1
                else:
                    pred_labels[i.item()] = j[1].item()
                    count[i.item()] = 1
            for i in pred_labels:
                pred_labels[i] /= count[i]

        return pred_labels
    


def load_data(data,tokenizer,num_document_graphs,all_amr_graphs):
    data = pd.DataFrame(data)
    dataset = Dataset.from_pandas(data)

    def preprocess_factgraph(examples):
        result = preprocess_function(examples, tokenizer, num_document_graphs,all_amr_graphs)
        return result
    
    dataset = dataset.map(preprocess_factgraph, batched=True,
                        load_from_cache_file=False,
                        remove_columns=["graph_summary"],
                        num_proc=1)
    dataset.set_format(columns=["input_ids", "hyp_id", "attention_mask", "graphs", "graphs_attn",
                                 "mask_graph", "edge_index", "edge_type"]) 
    return dataset    

def evaluate(data,fg_model,fg_tokenizer,num_sents,summ_location,amr_graphs):
    new_data = process_amr(data,summ_location)
    for example in new_data:
        example = dict(example)
    test_data = load_data(new_data,fg_tokenizer,num_sents,amr_graphs)
    pred = test(fg_model, test_data,num_sents)
    return pred
