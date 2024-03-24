# Configure datasets using for benchmarking of aging clock models.
DATASETS_PREFIX = '/tank/projects/computage/benchmarking/'

datasets_config = {
                    'GSE52588':{'path':f'{DATASETS_PREFIX}GSE52588.pkl.gz',
                               'condition':'DS',
                               'test':'AAP'
                               },
                    'GSE42861':{'path':f'{DATASETS_PREFIX}GSE42861.pkl.gz',
                               'condition':'Rheumatoid arthritis',
                               'test':'AAP'
                               },
                    'GSE59685':{'path':f'{DATASETS_PREFIX}GSE59685.pkl.gz',
                               'condition':'AD',
                               'test':'AAP'
                               },
                    'GSE80970':{'path':f'{DATASETS_PREFIX}GSE80970.pkl.gz',
                               'condition':'AD',
                               'test':'AAP'
                               },         
                    
                    #Immune system diseases
                    'GSE87640':{'path':f'{DATASETS_PREFIX}GSE87640.pkl.gz',
                               'condition':'IBD',
                               'test':'AAP'
                               },   
                    'GSE87648':{'path':f'{DATASETS_PREFIX}GSE87648.pkl.gz',
                               'condition':'IBD',
                               'test':'AAP'
                               },
                    'GSE32148':{'path':f'{DATASETS_PREFIX}GSE32148.pkl.gz',
                               'condition':'IBD',
                               'test':'AAP'
                               },    
                    'GSE53840':{'path':f'{DATASETS_PREFIX}GSE53840.pkl.gz',
                               'condition':'HIV',
                               'test':'AA0'
                               },   
                    
                    #Cardiovascular diseases                                         
                    'GSE157131':{'path':f'{DATASETS_PREFIX}GSE157131.pkl.gz',
                               'condition':'HT',
                               'test':'AAP'
                               },  
                    'GSE84395':{'path':f'{DATASETS_PREFIX}GSE84395.pkl.gz',
                               'condition':'PAH',
                               'test':'AAP'
                               },
                    'GSE56046':{'path':f'{DATASETS_PREFIX}GSE56046.pkl.gz',
                               'condition':'AS',
                               'test':'AAP'
                               },
                    'GSE56581':{'path':f'{DATASETS_PREFIX}GSE56046.pkl.gz',
                               'condition':'AS',
                               'test':'AAP'
                               },
                    'GSE107143':{'path':f'{DATASETS_PREFIX}GSE107143.pkl.gz',
                               'condition':'AS',
                               'test':'AAP'
                               },
                    'GSE62867':{'path':f'{DATASETS_PREFIX}GSE62867.pkl.gz',
                               'condition':'IHD',
                               'test':'AA0'
                               },
                    'GSE69138':{'path':f'{DATASETS_PREFIX}GSE69138.pkl.gz',
                               'condition':'CVA',
                               'test':'AA0'
                               },
                    'GSE203399':{'path':f'{DATASETS_PREFIX}GSE203399.pkl.gz',
                               'condition':'CVA',
                               'test':'AA0'
                               },
                    'GSE109096':{'path':f'{DATASETS_PREFIX}GSE109096.pkl.gz',
                               'condition':'HF',
                               'test':'AA0'
                               },
                    'GSE197670':{'path':f'{DATASETS_PREFIX}GSE197670.pkl.gz',
                               'condition':'HF',
                               'test':'AA0'
                               },
                    'GSE46394':{'path':f'{DATASETS_PREFIX}GSE46394.pkl.gz', #multiple points per human
                               'condition':'AS',
                               'test':'AAP'
                               },                               
                    
                    #Metabolic disorders
                    'GSE38291':{'path':f'{DATASETS_PREFIX}GSE38291.pkl.gz',
                               'condition':'T2D',
                               'test':'AAP'
                               },
                    'GSE49909':{'path':f'{DATASETS_PREFIX}GSE49909.pkl.gz', #TODO: process 2 conditions
                               'condition':'OBS',
                               'test':'AAP'
                               },
                    'GSE50498':{'path':f'{DATASETS_PREFIX}GSE50498.pkl.gz',
                               'condition':'OBS',
                               'test':'AAP'
                               },      
                    'GSE73103':{'path':f'{DATASETS_PREFIX}GSE73103.pkl.gz',
                               'condition':'OBS',
                               'test':'AAP'
                               },   
                    'GSE222595':{'path':f'{DATASETS_PREFIX}GSE222595.pkl.gz',
                               'condition':'OBS',
                               'test':'AAP'
                               },
                    'GSE48325':{'path':f'{DATASETS_PREFIX}GSE48325.pkl.gz', #TODO: process multiple conditions
                               'condition':'OBS', 
                               'test':'AAP'
                               },    
                    'GSE61256':{'path':f'{DATASETS_PREFIX}GSE48325.pkl.gz', #TODO: process multiple conditions
                               'condition':'OBS', 
                               'test':'AAP'
                               },        
                    'GSE62003':{'path':f'{DATASETS_PREFIX}GSE62003.pkl.gz', 
                               'condition':'T2D', 
                               'test':'AA0'
                               },                   
                   }
