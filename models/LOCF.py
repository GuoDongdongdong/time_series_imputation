from pypots.imputation import LOCF


class Model() :
    def __init__(self, args):
        super().__init__(args)
        self.model = LOCF(
            first_step_imputation=args.first_step_imputation
        )

    def impute(self, test_dataset : dict):
        return self.model.impute(test_dataset)