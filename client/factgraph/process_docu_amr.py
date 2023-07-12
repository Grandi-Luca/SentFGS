from amr_utils.amr_readers import AMR_Reader
import json
from factgraph.process_utils import simplify_amr_nopar
import collections
from factgraph.preprocess import align_triples,generate_edge_tensors

def process(amr_graph_path,length,tokenizer):
        reader = AMR_Reader()
        docu_sent_amr = []
        for i in range(length):
            #get sentences amr
            path = f'{amr_graph_path}/{i}-input_text.amr.txt'
            amrs = reader.load(path, remove_wiki=True)
            docu_sent_amr.extend(amrs)
        #all sentence amr
        amr_graphs = {}
        max_seq_length = 300
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        for amr in docu_sent_amr:
            try:
                graph = amr.graph_string()
                graph_simple, triples = simplify_amr_nopar(graph)
                # graph_dict = {}
                graph_simple = ' '.join(graph_simple)
                # graph_dict['amr_simple'] = graph_simple
                # graph_dict['triples'] = json.dumps(triples)
                # amr_graphs[amr.metadata['snt']] = graph_dict
                
                amr_processed_graph = {}
                doc_graph_string_tok = tokenizer.tokenize("[CLS] " + graph_simple + " [SEP]")
                # triples = process_triples(triples)
                all_triples = align_triples(doc_graph_string_tok, "[CLS] " + graph_simple + " [SEP]", triples, 1)
                edge_index, edge_types = generate_edge_tensors(all_triples, max_seq_length)

                input_ids_graph = tokenizer.convert_tokens_to_ids(doc_graph_string_tok)
                assert len(doc_graph_string_tok) == len(input_ids_graph), "length mismatched"
                padding_length_a = max_seq_length - len(doc_graph_string_tok)
                input_ids_graph = input_ids_graph + ([pad_token] * padding_length_a)
                graph_attention_mask = [1] * len(doc_graph_string_tok) + ([0] * padding_length_a)

                input_ids_graph = input_ids_graph[:max_seq_length]
                graph_attention_mask = graph_attention_mask[:max_seq_length]
                
                amr_processed_graph["enc_graphs"] = input_ids_graph
                amr_processed_graph["enc_graphs_attn"] = graph_attention_mask
                amr_processed_graph["graph_edge_index"] = edge_index
                amr_processed_graph["graph_edge_type"] = edge_types
                amr_graphs[amr.metadata['snt']] = amr_processed_graph

            except Exception as e:
                print(e)

        return amr_graphs
