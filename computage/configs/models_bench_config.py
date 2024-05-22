models_config = {
    "in_library":{
        'HorvathV1':{'imputation':common_imputation},
        'Hannum':{'imputation':common_imputation},
        'Lin':{'imputation':common_imputation},
        'PhenoAgeV1':{'imputation':common_imputation},
        'YingCausAge':{'imputation':common_imputation},
        'YingDamAge':{'imputation':common_imputation},
        'YingAdaptAge':{'imputation':common_imputation},
        'HorvathV2':{'imputation':common_imputation},
        'PhenoAgeV2':{'imputation':common_imputation},
		'VidalBralo':{'imputation':common_imputation},
		'Zhang19_EN':{'imputation':common_imputation},
        'GrimAgeV1':{'imputation':common_imputation},
        'GrimAgeV2':{'imputation':common_imputation},
    },
    #each model should have `path` in its dict values (see example)
    #each model should be stored in pickle (.pkl) format
    "new_models":{
        #'my_new_model_name': {'path':/path/to/model.pkl}
    }
}