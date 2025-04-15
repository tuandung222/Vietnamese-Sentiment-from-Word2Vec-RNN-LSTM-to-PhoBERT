# import common library
import wandb, datetime, pickle

# Model for experiment
from models.phow2vec import PhoW2VecWrapper
from models.cnn import CNNClassifier
from models.lstm import LSTMClassifier
from models.hybrid_cnn_lstm import HybridClassifer
from models.hf_wrapper import HuggingFaceModelWrapper

# Data Augmentation
from data.vietnamese_eda import VietnameseEDATransform
from data.dataset_builder import DatasetBuilder

# Train engine and evaluate engine
# train_evaluate_pipeline: train on train set, validate during training on validation set, evaluate on test set
from engine import MyTrainer, TraininingConfig, train_evaluate_pipeline

# Visualization Utils
from visualization import show_compared_test_table_prettytable, draw_training_history


def set_deterministic_seed(seed: int) -> None:
    import random, torch, numpy as np

    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def main():
    set_deterministic_seed(42)
    ##### Prepare Dataset
    train_augment_transform = VietnameseEDATransform()
    dataset_builder = DatasetBuilder(train_augment_transform=train_augment_transform)
    train_set, val_set, test_set = dataset_builder.build()

    ##### Build models for experiment
    word2vec_model = PhoW2VecWrapper(max_length=64)
    ## 1. CNN model
    cnn_model = CNNClassifier(
        word2vec_model=word2vec_model,
        input_dim=300,
        num_filters=300,
        filter_sizes=[3, 4, 5, 6, 7, 8],
        output_dim=3,
        dropout=0.3,
    )
    ## 2. LSTM model
    lstm_model = LSTMClassifier(
        word2vec_model=word2vec_model,
        input_dim=300,
        hidden_dims=[384, 384],
        output_dim=3,
        n_layers=2,
        bidirectional=True,
        dropout=0.3,
    )
    ## 3. Hybrid model CNN + LSTM
    crnn_model = HybridClassifer(
        word2vec_model=word2vec_model,
        input_dim=300,
        lstm_hidden_dim=384,
        dropout=0.3,
        cnn_num_filters=300,
        cnn_filter_sizes=[3, 4, 5, 6, 7, 8],
    )
    ## 4. PhoBERT model
    phobert_model = HuggingFaceModelWrapper(
        model_name="vinai/phobert-base-v2", num_classes=3, max_length=64
    )

    ##### Build train config and logger
    # config 1: for model relying on static word embedding
    training_config_1 = TraininingConfig(
        max_epochs=128,
        max_patience=12,
        batch_size=1024,
        lr=7.5e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.995),
        output_dir="checkpoints",
    )
    # for phobert model: low learning rate
    training_config_2 = TraininingConfig(
        max_epochs=128,
        max_patience=12,
        batch_size=512,
        lr=1e-5,
        weight_decay=1e-1,
        betas=(0.9, 0.995),
        output_dir="checkpoints",
    )
    list_configs = [training_config_2] + [training_config_1] * 3

    ##### Train and validate on train/val, then load best model to evaluate on test set
    model_names = ["phobert", "cnn", "lstm", "hybrid_crnn"]
    result_objects = {}
    for model, training_config, model_name in zip(
        [phobert_model, cnn_model, lstm_model, crnn_model], list_configs, model_names
    ):
        model.to("cuda")
        logger = wandb.init(
            anonymous="allow",  # comment if you want to log in
            project="NLP_Project_241",
            group="finetune_vlsp_2016",
            name=str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-"))
            + f"- finetune_vlsp - {model_name}",
            config=training_config,
            notes=model_name,
            tags=["<finetune>", "<vlsp>", model_name],
        )
        result_object = train_evaluate_pipeline(
            config=training_config,
            model=model,
            train_dataset=train_set,
            val_dataset=val_set,
            test_dataset=test_set,
            collate_fn=dataset_builder.collate_fn,
            logger=logger,
            device="cuda",
            num_workers=8,
        )
        model.to("cpu")
        result_objects.update({model_name: result_object})
        logger.finish()
        # result_objects {"training_result": ..., "test_result": ...}

        ##### Save result objects
        with open("result_objects.pkl", "wb") as f:
            pickle.dump(result_objects, f)

    ##### Visualize and compare results
    # 1. Show compared test table
    show_compared_test_table_prettytable(result_objects)
    # 2. Draw training history
    draw_training_history(result_objects)


if __name__ == "__main__":
    main()
