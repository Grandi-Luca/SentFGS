import factgraph.augmentation_ops as ops
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_sents(data, num_sents):
    sent_op = ops.SelectSentencesScore()
    data = apply_transformation_parallel(data, sent_op, apply_transformation, num_sents, 1000, 5)
    return data

def apply_transformation(data, operation, num_sents, idx_process):
    for idx, example in enumerate(data):
        try:
            new_example = operation.transform(example, number_sents=num_sents)
            if new_example:
                data[idx] = new_example
        except Exception as e:
            print("Caught exception:", e)
    return data


def apply_transformation_parallel(data, operation, transformation, num_sents, size_chunks, workers):
    # apply each chunks of data in parallel to execute the transformation
    data_list = list(chunks(data, size_chunks))
    set_start_method('spawn', force=True)
    final_datapoints = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for idx, data in enumerate(data_list):
            job = executor.submit(transformation, data, operation, num_sents, idx)
            futures[job] = idx

        for job in as_completed(futures):
            datapoint = job.result()
            r = futures[job]
            final_datapoints.extend(datapoint)
            del futures[job]
    return final_datapoints