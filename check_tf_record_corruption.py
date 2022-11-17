import tensorflow as tf
import glob
import sys
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# "quickdraw" not in fname and "ilsvrc" not in fname:
assets_glob = "../../../assets/data/tf2/records/{}/*.tfrecords"
records = list(sorted(glob.glob(assets_glob.format("ilsvrc_2012"), recursive=True)))
# list(sorted(glob.glob(assets_glob.format("quickdraw"), recursive=True)))
print(f"Found {len(records)} TFRecords")
n = 0
corrupted = []
for i, fname in enumerate(tqdm(records)):
    try:
        n += sum([1 for _ in tf.compat.v1.io.tf_record_iterator(fname)])
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as ex:
        with logging_redirect_tqdm():
            LOG.info(f"Error corrupted TFRecord@ {i}: {fname} ({type(ex).__name__}, {ex.args})")
        corrupted.append((i, fname))
    if i != 0 and i % 100 == 0:
        with logging_redirect_tqdm():
            LOG.info(f"{n} uncorrupted images processed")
print(f"{n} uncorrupted images found")
for i, fname in corrupted:
    print(f"Error corrupted TFRecord@ {i}: {fname}")
