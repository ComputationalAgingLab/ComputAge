# Configure datasets using for benchmarking of aging clock models.
DATASETS_PREFIX = '/tank/projects/computage/benchmarking/'

datasets_config = {
                    'GSE52588':{'path':f'{DATASETS_PREFIX}GSE52588.pkl.gz',
                               'condition':'DS',
                               'test':'AA2'
                               },
                    'GSE42861':{'path':f'{DATASETS_PREFIX}GSE42861.pkl.gz',
                               'condition':'RA',
                               'test':'AA2'
                               },
                    'GSE59685':{'path':f'{DATASETS_PREFIX}GSE59685.pkl.gz',
                               'condition':'AD',
                               'test':'AA2'
                               },
                    'GSE80970':{'path':f'{DATASETS_PREFIX}GSE80970.pkl.gz',
                               'condition':'AD',
                               'test':'AA2'
                               },         
                    
                    #Immune system diseases
                    'GSE87640':{'path':f'{DATASETS_PREFIX}GSE87640.pkl.gz',
                               'condition':'IBD',
                               'test':'AA2'
                               },   
                    'GSE87648':{'path':f'{DATASETS_PREFIX}GSE87648.pkl.gz',
                               'condition':'IBD',
                               'test':'AA2'
                               },
                    'GSE32148':{'path':f'{DATASETS_PREFIX}GSE32148.pkl.gz',
                               'condition':'IBD',
                               'test':'AA2'
                               },    
                    'GSE53840':{'path':f'{DATASETS_PREFIX}GSE53840.pkl.gz',
                               'condition':'HIV',
                               'test':'AA1'
                               },   
                    
                    #Cardiovascular diseases                                         
                    'GSE157131':{'path':f'{DATASETS_PREFIX}GSE157131.pkl.gz',
                               'condition':'HT',
                               'test':'AA2'
                               },  
                    'GSE84395':{'path':f'{DATASETS_PREFIX}GSE84395.pkl.gz',
                               'condition':'PAH',
                               'test':'AA2'
                               },
                    'GSE56046':{'path':f'{DATASETS_PREFIX}GSE56046.pkl.gz',
                               'condition':'AS',
                               'test':'AA2'
                               },
                    'GSE56581':{'path':f'{DATASETS_PREFIX}GSE56581.pkl.gz',
                               'condition':'AS',
                               'test':'AA2'
                               },
                    'GSE107143':{'path':f'{DATASETS_PREFIX}GSE107143.pkl.gz',
                               'condition':'AS',
                               'test':'AA2'
                               },
                    'GSE62867':{'path':f'{DATASETS_PREFIX}GSE62867.pkl.gz',
                               'condition':'IHD',
                               'test':'AA1'
                               },
                    'GSE69138':{'path':f'{DATASETS_PREFIX}GSE69138.pkl.gz',
                               'condition':'CVA',
                               'test':'AA1'
                               },
                    'GSE203399':{'path':f'{DATASETS_PREFIX}GSE203399.pkl.gz',
                               'condition':'CVA',
                               'test':'AA1'
                               },
                    'GSE109096':{'path':f'{DATASETS_PREFIX}GSE109096.pkl.gz',
                               'condition':'HF',
                               'test':'AA1'
                               },
                    'GSE197670':{'path':f'{DATASETS_PREFIX}GSE197670.pkl.gz',
                               'condition':'HF',
                               'test':'AA1'
                               },
                    'GSE46394':{'path':f'{DATASETS_PREFIX}GSE46394.pkl.gz', #multiple points per human
                               'condition':'AS',
                               'test':'AA2'
                               },                               
                    
                    #Metabolic disorders
                    'GSE38291':{'path':f'{DATASETS_PREFIX}GSE38291.pkl.gz',
                               'condition':'T2D',
                               'test':'AA2'
                               },
                    'GSE49909':{'path':f'{DATASETS_PREFIX}GSE49909.pkl.gz', #TODO: process 2 conditions
                               'condition':'OBS',
                               'test':'AA2'
                               },
                    'GSE50498':{'path':f'{DATASETS_PREFIX}GSE50498.pkl.gz',
                               'condition':'OBS',
                               'test':'AA2'
                               },      
                    'GSE73103':{'path':f'{DATASETS_PREFIX}GSE73103.pkl.gz',
                               'condition':'OBS',
                               'test':'AA2'
                               },   
                    'GSE222595':{'path':f'{DATASETS_PREFIX}GSE222595.pkl.gz',
                               'condition':'OBS',
                               'test':'AA2'
                               },
                    'GSE48325':{'path':f'{DATASETS_PREFIX}GSE48325.pkl.gz', #TODO: process multiple conditions
                               'condition':'OBS', 
                               'test':'AA2'
                               },    
                    'GSE61256':{'path':f'{DATASETS_PREFIX}GSE48325.pkl.gz', #TODO: process multiple conditions
                               'condition':'OBS', 
                               'test':'AA2'
                               },        
                    'GSE62003':{'path':f'{DATASETS_PREFIX}GSE62003.pkl.gz', 
                               'condition':'T2D', 
                               'test':'AA1'
                               },                   
                   }



datasets_config_bench1 = {
                    'GSE3214':{'path':f'{DATASETS_PREFIX}GSE3214.pkl.gz',
                               'condition':'IBD',
                               'test':'AA2'
                               },    

                   }



datasets_config_bench2 = {
                    'GSE32148':{'path':f'{DATASETS_PREFIX}GSE32148.pkl.gz',
                               'condition':'IBD',
                               'test':'AA2'
                               },    
                    'GSE56581':{'path':f'{DATASETS_PREFIX}GSE56581.pkl.gz',
                               'condition':'AS',
                               'test':'AA2'
                               },
                    'GSE87640':{'path':f'{DATASETS_PREFIX}GSE87640.pkl.gz',
                               'condition':'IBD',
                               'test':'AA2'
                               },   
                    'GSE67751':{'path':f'{DATASETS_PREFIX}GSE67751.pkl.gz',
                               'condition':'HIV',
                               'test':'AA2'
                               }, 
                    'GSE131989':{'path':f'{DATASETS_PREFIX}GSE131989.pkl.gz',
                               'condition':'RA',
                               'test':'AA2'
                               },          
                    'GSE134429':{'path':f'{DATASETS_PREFIX}GSE134429.pkl.gz',
                               'condition':'RA',
                               'test':'AA2'
                               },       
                    'GSE49909':{'path':f'{DATASETS_PREFIX}GSE49909.pkl.gz',
                               'condition':'OBS',
                               'test':'AA2'
                               },     
                    'GSE99624':{'path':f'{DATASETS_PREFIX}GSE99624.pkl.gz',
                               'condition':'OP',
                               'test':'AA2'
                               },  
                    # 'GSE100825':{'path':f'{DATASETS_PREFIX}GSE100825.pkl.gz',
                    #            'condition':'WS',
                    #            'test':'AA2'
                    #            },  #too few samples
                    'GSE131752':{'path':f'{DATASETS_PREFIX}GSE131752.pkl.gz',
                               'condition':'WS',
                               'test':'AA2'
                               },   
                    'GSE182991':{'path':f'{DATASETS_PREFIX}GSE182991.pkl.gz',
                               'condition':'HGPS',
                               'test':'AA2'
                               },    
                    'GSE214297':{'path':f'{DATASETS_PREFIX}GSE214297.pkl.gz',
                               'condition':'CGL',
                               'test':'AA2'
                               },   
                   }
