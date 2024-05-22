from matplotlib.colors import LinearSegmentedColormap

final_colors = {
                'CVD': '#EA999A',
                'ISD': '#B6D7A8',
                'MSD': '#D5A6BD',
                'MBD': '#FFE599',
                'NDD': '#B4A8D2',
                'RSD': '#A0C5E7',
                'PGS': '#A2C4C9',
                }

final_colors_light = {
                'CVD': '#f9e1e1',
                'ISD': '#e9f3e5',
                'MSD': '#f2e4eb',
                'MBD': '#fff7e0',
                'NDD': '#e9e5f2',
                'RSD': '#e3eef8',
                'PGS': '#e3edef',
                }

final_colors_dark = {
                'CVD': '#462e2e',
                'ISD': '#374032',
                'MSD': '#403239',
                'MBD': '#4c452e',
                'NDD': '#36323f',
                'RSD': '#303b45',
                'PGS': '#313b3c',
                }


pgs_cmap_light = LinearSegmentedColormap.from_list(
        name="pgs_cmap", colors=[final_colors['PGS'], final_colors_light['PGS']], N=256
    )
pgs_cmap_dark = LinearSegmentedColormap.from_list(
        name="pgs_cmap", colors=[final_colors['PGS'], final_colors_dark['PGS']], N=256
    )


rsd_cmap_light = LinearSegmentedColormap.from_list(
        name="rsd_cmap", colors=[final_colors['RSD'], final_colors_light['RSD']], N=256
    )
rsd_cmap_dark = LinearSegmentedColormap.from_list(
        name="rsd_cmap", colors=[final_colors['RSD'], final_colors_dark['RSD']], N=256
    )


ndd_cmap_light = LinearSegmentedColormap.from_list(
        name="ndd_cmap", colors=[final_colors['NDD'], final_colors_light['NDD']], N=256
    )
ndd_cmap_dark = LinearSegmentedColormap.from_list(
        name="ndd_cmap", colors=[final_colors['NDD'], final_colors_dark['NDD']], N=256
    )


mbd_cmap_light = LinearSegmentedColormap.from_list(
        name="mbd_cmap", colors=[final_colors['MBD'], final_colors_light['MBD']], N=256
    )
mbd_cmap_dark = LinearSegmentedColormap.from_list(
        name="mbd_cmap", colors=[final_colors['MBD'], final_colors_dark['MBD']], N=256
    )


cvd_cmap_light = LinearSegmentedColormap.from_list(
        name="cvd_cmap", colors=[final_colors['CVD'], final_colors_light['CVD']], N=256
    )
cvd_cmap_dark = LinearSegmentedColormap.from_list(
        name="cvd_cmap", colors=[final_colors['CVD'], final_colors_dark['CVD']], N=256
    )


isd_cmap_light = LinearSegmentedColormap.from_list(
        name="isd_cmap", colors=[final_colors['ISD'], final_colors_light['ISD']], N=256
    )
isd_cmap_dark = LinearSegmentedColormap.from_list(
        name="isd_cmap", colors=[final_colors['ISD'], final_colors_dark['ISD']], N=256
    )


msd_cmap_light = LinearSegmentedColormap.from_list(
        name="msd_cmap", colors=[final_colors['MSD'], final_colors_light['MSD']], N=256
    )
msd_cmap_dark = LinearSegmentedColormap.from_list(
        name="msd_cmap", colors=[final_colors['MSD'], final_colors_dark['MSD']], N=256
    )

class_light_cmap_dict = {
                'CVD': cvd_cmap_light,
                'ISD': isd_cmap_light,
                'MSD': msd_cmap_light,
                'MBD': mbd_cmap_light,
                'NDD': ndd_cmap_light,
                'RSD': rsd_cmap_light,
                'PGS': pgs_cmap_light,
                }

grays_cmap_light = LinearSegmentedColormap.from_list(
        name="grays_cmap", colors=['#D3D3D3', '#404040'], N=256
    )
