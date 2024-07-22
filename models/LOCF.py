from pypots.imputation import LOCF


class Model() :


    def __init__(self, args):
        super().__init__(args)
        self.model = LOCF(
            first_step_imputation=args.first_step_imputation
        )
