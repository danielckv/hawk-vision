from affine import Affine
from osgeo import gdal, osr
from algorithms.io.utils import load_image


class GeoTiffHandler:
    def __init__(self, path):
        self.path = path
        self.dataset = gdal.Open(path)
        self.metadata = self.dataset.GetMetadata()
        self.projection = self.dataset.GetProjection()
        self.geo_transform = self.dataset.GetGeoTransform()
        self.band = self.dataset.GetRasterBand(1)
        self.data = self.band.ReadAsArray()
        self.no_data_value = self.band.GetNoDataValue()
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.extent = self.get_extent()
        self.crs = self.get_crs()
        self.transform = self.get_transform()
        self.res = self.get_resolution()

        print(
            f"GeoTiffHandler: {self.path} {self.width}x{self.height} {self.extent} {self.crs} {self.transform} {self.res}")

    def get_image_pillow(self):
        return load_image(self.path)

    def get_extent(self):
        min_x = self.geo_transform[0]
        max_x = min_x + self.width * self.geo_transform[1]
        min_y = self.geo_transform[3] + self.height * self.geo_transform[5]
        max_y = self.geo_transform[3]
        return [min_x, min_y, max_x, max_y]

    def get_crs(self):
        crs = osr.SpatialReference()
        crs.ImportFromWkt(self.projection)
        return crs

    def get_transform(self):
        return Affine.from_gdal(*self.geo_transform)

    def get_resolution(self):
        return (self.geo_transform[1], -self.geo_transform[5])

    def get_value(self, x, y):
        return self.data[y, x]

    def get_values(self, x, y):
        return self.data[y, x]
