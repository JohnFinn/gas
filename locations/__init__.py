import pandas as pd
from pathlib import Path

class Coordinates:

    def __init__(self):
        self.countries: pd.DataFrame = pd.read_csv(Path(__file__).with_name('countries.csv'), sep='\t')
        self.cities: pd.DataFrame = pd.read_csv(Path(__file__).with_name('worldcities.csv'))

    def get_location(self, name: str) -> (float, float):
        data_point = self.countries[self.countries['name'] == name]
        if len(data_point) != 0:
            return tuple(data_point[['longitude', 'latitude']].values.reshape(2))
        data_point = self.cities[self.cities['city'] == name]
        if len(data_point) != 0:
            return tuple(data_point.iloc[0][['lng', 'lat']].values.reshape(2))
        data_point = self.cities[self.cities['city_ascii'] == name]
        if len(data_point) != 0:
            return tuple(data_point.iloc[0][['lng', 'lat']].values.reshape(2))
        return locations[name]

    def __getitem__(self, name):
        return self.get_location(name)

locations = {
 'Oltingue': (7.3914151, 47.4910127),
 'Zevenaar': (6.0789395, 51.9357252),
 'Biriatou': (-1.7431876, 43.3335124),
 'Wallbach': (10.40149001074598, 50.63469345),
 'Dravaszerdahely': (18.1635815, 45.8363381),
 'Kondratki': (23.9125019, 53.0180559),
 'Privalka': (23.966352, 53.9400472),
 'Brandov': (13.3907151, 50.632002),
 'Hilvarenbeek': (5.1379984, 51.4862924),
 'Thayngen': (8.7061689, 47.7470005),
 'Dragor': (12.6719953, 55.592487),
 'Fernana': (8.6957875, 36.6556629),
 'Tarifa': (-5.6048873, 36.0127749),
 'Gela': (14.2502445, 37.0664363),
 'Velke Kapusany': (22.07586275030526, 48.54143115),
 'Moravia': (16.514211, 49.140788),
 'Quevy (H)': (3.9439033, 50.3667226),
 'Baumgarten': (11.8710662, 53.8121231),
 'Hora svate Kateriny': (13.4369475, 50.6062324),
 'Hermanowice': (22.8149504, 49.7252258),
 'Waidhaus': (12.4965684, 49.6411504),
 'Bras': (5.818012, 49.987242),
 'Karksi': (25.58587410455435, 58.11455895),
 'Kipi': (22.045654102717084, 58.249517499999996),
 'Nybro': (15.828309119659192, 56.8369395),
 'Oberkappel': (13.7707406, 48.55284),
 'Velke Zlievce': (19.4559418, 48.1993446),
 'Sâkiai': (23.0395432, 54.9559617),
 'Kotlovka': (37.6170315, 55.6847011),
 'Turkgozu': (42.8191224, 41.5800821),
 'Laa': (-0.7254521, 43.4174248),
 'Kiskundorozsma': (20.0657989, 46.2744169),
 'Varska': (27.64257956235013, 57.954720550000005),
 'Zandvliet (L)': (4.3086817, 51.3599666),
 'Dornum': (7.4279332, 53.6469082),
 'Mazara del Vallo': (12.5886912, 37.6537292),
 'Korneti': (26.9479754, 57.5899347),
 'Kamminke': (14.2065532, 53.8684526),
 'Tuy': (-8.642214, 42.049343),
 'Lanzhot': (16.9669476, 48.7244326),
 'Budince': (22.1211139, 48.5454814),
 'BBL': (7.3875078, 46.946199),
 'Petange': (5.87971, 49.558174),
 'Moffat': (-3.441224, 55.333856),
 'Obbicht': (5.779748177430276, 51.02590395),
 'Tasnieres (L)': (2.507849, 47.2024211),
 'Tarvisio': (13.5783734, 46.5052624),
 'Winterswijk': (6.7224185, 51.9746432),
 'Alveringem': (2.7105025, 51.0119984),
 'Northern Offshore Gas Transport (NOGAT)': (4.7369111, 54.1465652),
 'Oude (L-Gas)': (6.417692, 51.897176),
 'Griespass': (8.3739756, 46.4537719),
 'Kieménai': (25.737118, 54.552414),
 's Gravenvoeren': (5.7619737, 50.7588019),
 'Murfeld': (15.4580055, 47.0231126),
 'Bocholtz': (5.9990800902879755, 50.81656985),
 'Jura': (5.783285726354901, 46.783362499999996),
 'Ellund': (9.3107737, 54.794602),
 'Mediesu Aurit': (23.1364319, 47.7895851),
 'Kiefersfelden': (12.188782, 47.6138408),
 'Blaregnies (H)': (3.8980429, 50.357777),
 'Interconnector': (3.1766494, 51.3236301),
 'Malkoclar': (27.018753, 42.0408566),
 'Beregdaróc': (22.5309278, 48.1970929),
 'Larrau': (-0.95593, 43.0183998),
 'Branice': (14.339536, 49.4024637),
 'Bizzarone Como': (8.942735, 45.834278),
 'Csanadpalota': (20.723398, 46.2440484),
 'Easington': (-1.0051337, 51.7871889),
 'Oude (H-Gas)': (6.417692, 51.897176),
 'Uberackern II': (12.8760256, 48.1931447),
 'Lasow': (22.260833, 50.9347238),
 'Nord Stream': (13.6572167, 54.1521234),
 'Wysokoje': (31.481861, 54.315548),
 'Obergailbach': (7.2186154, 49.11864),
 'Zandvliet (H)': (4.3086817, 51.3599666),
 'Uberackern': (12.8760256, 48.1931447),
 'Dinxperlo (RWE)': (6.485727, 51.868411),
 'Dogubajazit': (44.054867, 39.558545),
 'Drozdowicze': (19.95264, 50.051816),
 'Emden (EPT1)': (7.189162, 53.360922),
 'Emden (NPT)': (7.189162, 53.360922),
 'Eynatten (EON/RWE)': (6.081911, 50.692849),
 'Eynatten (Sum)': (6.081911, 50.692849),
 'Eynatten (Wingas)': (6.081911, 50.692849),
 'Haanrade (RWE)': (6.074921, 50.881906),
 'Interfield transfer from Blane to Ula': (-3.330949, 56.020204),
 'Jidilovo': (22.399953, 42.223454),
 'Mallnow (Frankfurt am Oder)': (14.517429, 52.337156),
 'Not Elsewhere Specified': (0.0, 0.0),
 'Ruggel': (9.531786, 47.249345),
 'St Fergus (Alvheim and Edvard Greig)': (-1.851213, 57.572238),
 'St Fergus (Statfjord)': (-1.851213, 57.572238),
 'St Fergus (Vesterled)': (-1.851213, 57.572238),
 'Teesside via CATS': (-1.203439, 54.582747),
 'Tegelen (OGE)': (6.157514, 51.332252),
 'Tietierowka': (12.555874, 53.774088),
 'Vlieghuis (RWE)': (6.835622, 52.660973),
 'WestGasTransport (WGT) Pipeline': (3.954397, 53.084024),
 'Zeebrugge (ZPT)': (3.210252, 51.324493),
 'Zelzate (GTS)': (3.793242, 51.206748),
 'Zelzate (Zebra)': (3.79324, 51.2067482)
}