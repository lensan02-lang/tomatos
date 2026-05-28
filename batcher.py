import itertools

import h5py
import jax.numpy as jnp
import numpy as np


def get_generator(config, split):
    with h5py.File(config.preprocess_files[split], "r") as f:
        ds = f["stack"]
        print(ds.shape)        # Gesamtform
        print("chunks:",ds.chunks)       # Chunk-Größe
        chunks = list(ds.iter_chunks())
        print(f"Anzahl Chunks: {len(chunks)}")
        #print(f"Kombinationen (r=6): {len(list(itertools.combinations(chunks, r=6)))}")
        # this makes all non-repetitive combinations of batches, for r=2
        # [1,2,3] --> [[1,2], [1,3], [2,3]]
        # load the combinations together to avoid tiny rest batch, more data
        # variability
        # r slice combinations of the memory layout on disk
        batch_combi = list(
            itertools.combinations(list(ds.iter_chunks()), r=config.n_chunk_combine)
        )
        while True:
            for batch in batch_combi:
                # shape is (n_sample_sys, n_events, n_vars)
                batch = jnp.concatenate([ds[b] for b in batch], axis=1)
                # scale to total events
                batch_events = batch.shape[1]
                batch_sf = config.preprocess_md[split]["events"] / batch_events
                # this is an array of sample_sys size
                scale_factor = jnp.array(
                    np.array(config.preprocess_md[split]["scale_factor"]) * batch_sf
                )

                yield batch, scale_factor
            np.random.shuffle(batch_combi)