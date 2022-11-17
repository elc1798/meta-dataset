SHELL = /bin/bash

FULL_DATA_DIR := ../../../assets/data/full
DATA_SPLITS_DIR := meta_dataset/dataset_conversion/splits
DATA_TF2_RECORDS_DIR := ../../../assets/data/tf2/records
CHECKPOINTS_DIR := ../../../assets/checkpoints

ifeq ($(DATASET), imagenet)
    DATASET_ID := ilsvrc_2012
    DATA_SUBDIR := ILSVRC2012_img_train
else ifeq ($(DATASET), omniglot)
    DATASET_ID := omniglot
    DATA_SUBDIR := omniglot
else ifeq ($(DATASET), aircraft)
    DATASET_ID := aircraft
    DATA_SUBDIR := fgvc-aircraft-2013b/fgvc-aircraft-2013b
else ifeq ($(DATASET), cu_birds)
    DATASET_ID := cu_birds
    DATA_SUBDIR := CUB_200_2011/CUB_200_2011
else ifeq ($(DATASET), dtd)
    DATASET_ID := dtd
    DATA_SUBDIR := dtd/dtd
else ifeq ($(DATASET), quickdraw)
    DATASET_ID := quickdraw
    DATA_SUBDIR := quickdraw
else ifeq ($(DATASET), fungi)
    DATASET_ID := fungi
    DATA_SUBDIR := fungi
else ifeq ($(DATASET), vgg_flower)
    DATASET_ID := vgg_flower
    DATA_SUBDIR := vgg_flower
else ifeq ($(DATASET), traffic_sign)
    DATASET_ID := traffic_sign
    DATA_SUBDIR := GTSRB/GTSRB
else ifeq ($(DATASET), mscoco)
    DATASET_ID := mscoco
    DATA_SUBDIR := mscoco
endif

JQ_REMOVE_PATH := jq --sort-keys 'del(.path)'

.PHONY: preprocess_data
preprocess_data:
	python3 -m meta_dataset.dataset_conversion.convert_datasets_to_records \
		--dataset=$(DATASET_ID) \
		--$(DATASET_ID)_data_root=$(FULL_DATA_DIR)/$(DATA_SUBDIR) \
		--splits=$(DATA_SPLITS_DIR) \
		--records_root=$(DATA_TF2_RECORDS_DIR)

.PHONY: verify_dataspec
verify_dataspec:
	diff <($(JQ_REMOVE_PATH) $(DATA_TF2_RECORDS_DIR)/$(DATASET_ID)/dataset_spec.json) <($(JQ_REMOVE_PATH) meta_dataset/dataset_conversion/dataset_specs/$(DATASET_ID)_dataset_spec.json)


.PHONY: repro_flute
repro_flute:
	PYTHONPATH=${PYTHONPATH}:../task_adaptation python3 -m meta_dataset.train_flute \
		--train_checkpoint_dir=$(CHECKPOINTS_DIR)/flute_repro \
		--summary_dir=$(CHECKPOINTS_DIR)/flute_repro \
		--records_root_dir=$(DATA_TF2_RECORDS_DIR) \
		--alsologtostderr \
		--gin_config=meta_dataset/learn/gin/default/flute.gin \
		--gin_bindings="Trainer_flute.experiment_name='flute'"


.PHONY: flailnet
flailnet:
	PYTHONPATH=${PYTHONPATH}:../task_adaptation python3 -m meta_dataset.train_flute \
		--train_checkpoint_dir=$(CHECKPOINTS_DIR)/flailnet1 \
		--summary_dir=$(CHECKPOINTS_DIR)/flailnet1 \
		--records_root_dir=$(DATA_TF2_RECORDS_DIR) \
		--alsologtostderr \
		--gin_config=meta_dataset/learn/gin/default/flailnet.gin \
		--gin_bindings="Trainer_flute.experiment_name='flailnet'"
