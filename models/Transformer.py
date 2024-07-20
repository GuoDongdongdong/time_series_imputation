from models.base import BaseModel


from pypots.imputation import Transformer


class Model(BaseModel) :
    def __init__(self, args):
        super().__init__(args)
        self.model = Transformer(
            n_steps = args.lookback_len,
            n_features = args.features,
            n_layers = args.n_layers,
            d_model = args.d_model,
            d_inner = args.d_inner,
            n_heads = args.n_heads,
            d_k = args.d_k,
            d_v = args.d_v,
            dropout = args.dropout,
            attn_dropout = args.dropout,
            ORT_weight= args.ORT_weight,
            MIT_weight= args.MIT_weight,
            batch_size= args.batch_size,
            epochs= args.epochs,
            patience= args.patience,
            num_workers= args.num_workers,
            saving_path= args.saving_path,
            model_saving_strategy= args.model_saving_strategy
        )