import os
import time
import inspect

import util


def iterate_dataset(dl, ds):

    device = ds.device
    bs = dl.batch_size
    k = dl.sampler.k
    n = dl.sampler.n
    m = dl.sampler.m

    util.section('Dataset Details')
    util.tabulate({
        'Name': ds.__name__,
        'Device': device,
        'Total Classes': len(ds.classes()),
        'Total Images': len(ds.examples(None)),
        'Images per Task': k * (n + m),
        'Batch Size': bs,
        'k': k,
        'n': n,
        'm': m,
    })

    util.section('Iterating Dataset')

    start = time.perf_counter()

    i = 0
    for item in iter(dl):
        i += 1
        # Check to see if item is nested
        if not isinstance(item[0], list):
            item = [item]
        for j, (s, q) in enumerate(item):
            print(j, ': ', tuple(s.shape), tuple(q.shape))
        print('')

    stop = time.perf_counter()
    runtime = stop - start

    n_s = i * bs * k * n
    n_q = i * bs * k * m

    util.section('Results')
    util.tabulate({
        'Total Time': runtime,
        'Batch per Second': i / runtime,
        'Task per Second': (i * bs) / runtime,
        'Images per Second': (n_s + n_q) / runtime,
        'Total Batches': i,
        'Total Tasks': i * bs,
        'Total Support Images': n_s,
        'Total Query Images': n_q,
        'Total Images': n_s + n_q,
        'Percent Permuted': 100 * (n_s + n_q) / len(ds.examples(None)),
    })
