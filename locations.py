import pandas as pd

class Coordinates:

    def __init__(self):
        self.countries: pd.DataFrame = pd.read_csv('countries.csv', sep='\t')
        self.cities: pd.DataFrame = pd.read_csv('worldcities.csv')

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

locations = {
 'Oltingue': (47.4910127, 7.3914151),
 'Zevenaar': (51.9357252, 6.0789395),
 'Biriatou': (43.3335124, -1.7431876),
 'Wallbach': (50.63469345, 10.40149001074598),
 'Dravaszerdahely': (45.8363381, 18.1635815),
 'Kondratki': (53.0180559, 23.9125019),
 'Privalka': (53.9400472, 23.966352),
 'Brandov': (50.632002, 13.3907151),
 'Hilvarenbeek': (51.4862924, 5.1379984),
 'Thayngen': (47.7470005, 8.7061689),
 'Dragor': (55.592487, 12.6719953),
 'Fernana': (36.6556629, 8.6957875),
 'Tarifa': (36.0127749, -5.6048873),
 'Gela': (37.0664363, 14.2502445),
 'Velke Kapusany': (48.54143115, 22.07586275030526),
 'Moravia': (49.140788, 16.514211),
 'Quevy (H)': (50.3667226, 3.9439033),
 'Baumgarten': (53.8121231, 11.8710662),
 'Hora svate Kateriny': (50.6062324, 13.4369475),
 'Hermanowice': (49.7252258, 22.8149504),
 'Waidhaus': (49.6411504, 12.4965684),
 'Bras': (49.987242, 5.818012),
 'Karksi': (58.11455895, 25.58587410455435),
 'Kipi': (58.249517499999996, 22.045654102717084),
 'Nybro': (56.8369395, 15.828309119659192),
 'Oberkappel': (48.55284, 13.7707406),
 'Velke Zlievce': (48.1993446, 19.4559418),
 'Sâkiai': (54.9559617, 23.0395432),
 'Kotlovka': (55.6847011, 37.6170315),
 'Turkgozu': (41.5800821, 42.8191224),
 'Laa': (43.4174248, -0.7254521),
 'Kiskundorozsma': (46.2744169, 20.0657989),
 'Varska': (57.954720550000005, 27.64257956235013),
 'Zandvliet (L)': (51.3599666, 4.3086817),
 'Dornum': (53.6469082, 7.4279332),
 'Mazara del Vallo': (37.6537292, 12.5886912),
 'Korneti': (57.5899347, 26.9479754),
 'Kamminke': (53.8684526, 14.2065532),
 'Tuy': (42.049343, -8.642214),
 'Lanzhot': (48.7244326, 16.9669476),
 'Budince': (48.5454814, 22.1211139),
 'BBL': (46.946199, 7.3875078),
 'Petange': (49.558174, 5.87971),
 'Moffat': (55.333856, -3.441224),
 'Obbicht': (51.02590395, 5.779748177430276),
 'Tasnieres (L)': (47.2024211, 2.507849),
 'Tarvisio': (46.5052624, 13.5783734),
 'Winterswijk': (51.9746432, 6.7224185),
 'Alveringem': (51.0119984, 2.7105025),
 'Northern Offshore Gas Transport (NOGAT)': (54.1465652, 4.7369111),
 'Oude (L-Gas)': (51.897176, 6.417692),
 'Griespass': (46.4537719, 8.3739756),
 'Kieménai': (54.552414, 25.737118),
 's Gravenvoeren': (50.7588019, 5.7619737),
 'Murfeld': (47.0231126, 15.4580055),
 'Bocholtz': (50.81656985, 5.9990800902879755),
 'Jura': (46.783362499999996, 5.783285726354901),
 'Ellund': (54.794602, 9.3107737),
 'Mediesu Aurit': (47.7895851, 23.1364319),
 'Kiefersfelden': (47.6138408, 12.188782),
 'Blaregnies (H)': (50.357777, 3.8980429),
 'Interconnector': (51.3236301, 3.1766494),
 'Malkoclar': (42.0408566, 27.018753),
 'Beregdaróc': (48.1970929, 22.5309278),
 'Larrau': (43.0183998, -0.95593),
 'Branice': (49.4024637, 14.339536),
 'Bizzarone Como': (45.834278, 8.942735),
 'Csanadpalota': (46.2440484, 20.723398),
 'Easington': (51.7871889, -1.0051337),
 'Oude (H-Gas)': (51.897176, 6.417692),
 'Uberackern II': (48.1931447, 12.8760256),
 'Lasow': (50.9347238, 22.260833),
 'Nord Stream': (54.1521234, 13.6572167),
 'Wysokoje': (54.315548, 31.481861),
 'Obergailbach': (49.11864, 7.2186154),
 'Zandvliet (H)': (51.3599666, 4.3086817),
 'Uberackern': (48.1931447, 12.8760256),
 'Dinxperlo (RWE)': (51.868411, 6.485727),
 'Dogubajazit': (39.558545, 44.054867),
 'Drozdowicze': (50.051816, 19.95264),
 'Emden (EPT1)': (53.360922, 7.189162),
 'Emden (NPT)': (53.360922, 7.189162),
 'Eynatten (EON/RWE)': (50.692849, 6.081911),
 'Eynatten (Sum)': (50.692849, 6.081911),
 'Eynatten (Wingas)': (50.692849, 6.081911),
 'Haanrade (RWE)': (50.881906, 6.074921),
 'Interfield transfer from Blane to Ula': (56.020204, -3.330949),
 'Jidilovo': (42.223454, 22.399953),
 'Mallnow (Frankfurt am Oder)': (52.337156, 14.517429),
 'Not Elsewhere Specified': (0.0, 0.0),
 'Ruggel': (47.249345, 9.531786),
 'St Fergus (Alvheim and Edvard Greig)': (57.572238, -1.851213),
 'St Fergus (Statfjord)': (57.572238, -1.851213),
 'St Fergus (Vesterled)': (57.572238, -1.851213),
 'Teesside via CATS': (54.582747, -1.203439),
 'Tegelen (OGE)': (51.332252, 6.157514),
 'Tietierowka': (53.774088, 12.555874),
 'Vlieghuis (RWE)': (52.660973, 6.835622),
 'WestGasTransport (WGT) Pipeline': (53.084024, 3.954397),
 'Zeebrugge (ZPT)': (51.324493, 3.210252),
 'Zelzate (GTS)': (51.206748, 3.793242),
 'Zelzate (Zebra)': (51.206748, 3.793242)
}