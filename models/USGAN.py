from models.base import BaseModel


from pypots.imputation import USGAN


class Model(BaseModel) :


    def __init__(self, args):
        super().__init__(args)
        self.model = USGAN(
            n_steps= args.lookback_len,
            n_features= args.features,
            rnn_hidden_size= args.rnn_hidden_size,
            lambda_mse= args.lambda_mse,
            hint_rate= args.hint_rate,
            dropout= args.dropout,
            G_steps= args.G_steps,
            D_steps= args.D_steps,
            batch_size= args.batch_size,
            epochs= args.epochs,
            patience= args.patience,
            num_workers= args.num_workers,
            saving_path= args.saving_path,
            model_saving_strategy= args.model_saving_strategy
        )
