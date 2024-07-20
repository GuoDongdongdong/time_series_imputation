from pypots.imputation.saits.core import _SAITS




class Model(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = _SAITS(
            n_layers=args.n_layers,
            n_steps=args.seq_len,
            n_features=args.features,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_k=args.d_k,
            d_v=args.d_v,
            d_ffn=args.d_ffn,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            diagonal_attention_mask=args.diagonal_attention_mask,
            ORT_weight=args.ORT_weight,
            MIT_weight=args.MIT_weight,
        )