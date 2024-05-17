import os
import numpy as np
from computage.settings import ROOTDIR

#function for clock file retrieval
def get_clock_file(filename):
    clock_file_path = os.path.join(ROOTDIR, "models_library/clocks", filename)  
    return clock_file_path

#Horvath-specific ELU-like transform
def anti_trafo(x, adult_age=20):
    y = np.where(
        x < 0, (1 + adult_age) * np.exp(x) - 1, (1 + adult_age) * x + adult_age
    )
    return y

#identity transform if not given other
def identity(x):
    return x

### model definitions are taken from biolearn: https://github.com/bio-learn/biolearn/blob/master/ ###

model_definitions = {
    "HorvathV1": {
        "year": 2013,
        "species": "Human",
        "tissue": "Multi-tissue",
        "source": "https://genomebiology.biomedcentral.com/articles/10.1186/gb-2013-14-10-r115",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Horvath1.csv",
            "transform": lambda sum: anti_trafo(sum + 0.696),
        },
    },
    "Hannum": {
        "year": 2013,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.sciencedirect.com/science/article/pii/S1097276512008933",
        "output": "Age (Years)",
        "model": {"type": "LinearMethylationModel", "file": "Hannum.csv"},
    },
    "Lin": {
        "year": 2016,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.aging-us.com/article/100908/text",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Lin.csv",
            "transform": lambda sum: sum + 12.2169841,
        },
    },
    "PhenoAgeV1": {
        "year": 2018,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.aging-us.com/article/101414/text",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "PhenoAge.csv",
            "transform": lambda sum: sum + 60.664,
        },
    },
    "YingCausAge": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.biorxiv.org/content/10.1101/2022.10.07.511382v2",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "YingCausAge.csv",
            "transform": lambda sum: sum + 86.80816381,
        },
    },
    "YingDamAge": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.biorxiv.org/content/10.1101/2022.10.07.511382v2",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "YingDamAge.csv",
            "transform": lambda sum: sum + 543.4315887,
        },
    },
    "YingAdaptAge": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.biorxiv.org/content/10.1101/2022.10.07.511382v2",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "YingAdaptAge.csv",
            "transform": lambda sum: sum - 511.9742762,
        },
    },
    "HorvathV2": {
        "year": 2018,
        "species": "Human",
        "tissue": "Skin + blood",
        "source": "https://www.aging-us.com/article/101508/text",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Horvath2.csv",
            "transform": lambda sum: anti_trafo(sum - 0.447119319),
        },
    },
    "PEDBE": {
        "year": 2019,
        "species": "Human",
        "tissue": "Buccal",
        "source": "https://www.pnas.org/doi/10.1073/pnas.1820843116",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "PEDBE.csv",
            "transform": lambda sum: anti_trafo(sum - 2.1),
        },
    },
    "Zhang17": {
        "year": 2017,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.nature.com/articles/ncomms14617",
        "output": "Mortality Risk",
        "model": {
            "type": "LinearMethylationModel", 
            "file": "Zhang17.csv"},
    },
    "Zhang19_EN": {
        "year": 2019,
        "species": "Human",
        "tissue": "Blood|Saliva",
        "source": "https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-019-0667-1",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Zhang19_EN.csv",
            "transform": lambda sum: sum + 65.79295,
            }
        },
    "Zhang19_BLUP": {
        "year": 2019,
        "species": "Human",
        "tissue": "Blood|Saliva",
        "source": "https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-019-0667-1",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Zhang19_BLUP.csv",
            "transform": lambda sum: sum + 91.15396,
        },
    },
    # "DunedinPoAm38": {
    #     "year": 2020,
    #     "species": "Human",
    #     "tissue": "Blood",
    #     "source": "https://elifesciences.org/articles/54870#s2",
    #     "output": "Aging Rate (Years/Year)",
    #     "model": {
    #         "type": "LinearMethylationModel",
    #         "file": "DunedinPoAm38.csv",
    #         "transform": lambda sum: sum - 0.06929805,
    #     },
    # },
    # "DunedinPACE": {
    #     "year": 2022,
    #     "species": "Human",
    #     "tissue": "Blood",
    #     "source": "https://www.proquest.com/docview/2634411178",
    #     "output": "Aging Rate (Years/Year)",
    #     "model": {
    #         "type": "LinearMethylationModel",
    #         "file": "DunedinPACE.csv",
    #         "transform": lambda sum: sum - 1.949859,
    #         "preprocess": dunedin_pace_normalization,
    #         "default_imputation": "none",
    #     },
    # },
    "GrimAgeV1": {
        "year": 2019,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6366976/",
        "output": "Mortality Adjusted Age (Years)",
        "model": {"type": "GrimAgeModel", 
                  "file": "GrimAgeV1.csv"},
    },
    "GrimAgeV2": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9792204/",
        "output": "Mortality Adjusted Age (Years)",
        "model": {"type": "GrimAgeModel", 
                  "file": "GrimAgeV2.csv"},
    },
    "DNAmTL": {
        "year": 2019,
        "species": "Human",
        "tissue": "Blood, Adipose",
        "source": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6738410/",
        "output": "Telomere Length",
        "model": {
            "type": "LinearMethylationModel",
            "file": "DNAmTL.csv",
            "transform": lambda sum: sum - 7.924780053,
        },
    },
    "PhenoAgeV2": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "output": "Age (Years)",
        "source": "https://www.nature.com/articles/s43587-022-00248-2",
        "model": {
            "type": "LinearMethylationModel",
            "file": "HRSInCHPhenoAge.csv",
            "transform": lambda sum: sum + 52.8334080,
        },
    },
    "Knight": {
        "year": 2016,
        "species": "Human",
        "tissue": "Cord Blood",
        "source": "https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1068-z",
        "output": "Gestational Age",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Knight.csv",
            "transform": lambda sum: sum + 41.7,
        },
    },
    "LeeControl": {
        "year": 2019,
        "species": "Human",
        "tissue": "Placenta",
        "source": "https://www.aging-us.com/article/102049/text",
        "output": "Gestational Age",
        "model": {
            "type": "LinearMethylationModel",
            "file": "LeeControl.csv",
            "transform": lambda sum: sum + 13.06182,
        },
    },
    "LeeRefinedRobust": {
        "year": 2019,
        "species": "Human",
        "tissue": "Placenta",
        "source": "https://www.aging-us.com/article/102049/text",
        "output": "Gestational Age",
        "model": {
            "type": "LinearMethylationModel",
            "file": "LeeRefinedRobust.csv",
            "transform": lambda sum: sum + 30.74966,
        },
    },
    "LeeRobust": {
        "year": 2019,
        "species": "Human",
        "tissue": "Placenta",
        "source": "https://www.aging-us.com/article/102049/text",
        "output": "Gestational Age",
        "model": {
            "type": "LinearMethylationModel",
            "file": "LeeRobust.csv",
            "transform": lambda sum: sum + 24.99772,
        },
    },
    "VidalBralo": {
        "year": 2016,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2016.00126/full",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "VidalBralo.csv",
            "transform": lambda sum: sum + 84.7,
        },
    },
}
