models_config = {
    "in_library":{
        'Horvathv1':{},
        'Hannum':{},
        'Lin':{},
        'PhenoAge':{},
        'YingCausAge':{},
        'YingDamAge':{},
        'YingAdaptAge':{},
        'Horvathv2':{},
        'PEDBE':{},
        'HRSInCHPhenoAge':{},
    },
    #each model should have `path` in its dict values (see example)
    #each model should be stored in pickle (.pkl) format
    "new_models":{
        #'my_new_model_name': {'path':/path/to/model.pkl}
        
    }
}