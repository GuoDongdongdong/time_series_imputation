from models.base import BaseModel


from pypots.imputation import BRITS


class Model(BaseModel) :


    def __init__(self, args):
        super().__init__(args)
        self.model = BRITS(
            n_steps=args.lookback_len,
            n_features=args.features,
            rnn_hidden_size=args.rnn_hidden_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            saving_path=args.saving_path,
            model_saving_strategy=args.model_saving_strategy
        )
